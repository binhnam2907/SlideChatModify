# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from types import FunctionType
import json
import shutil
import tempfile
import time
import re
import glob
import sys
import math
from pathlib import Path
from typing import List, Dict, Any
from unicodedata import normalize
from multiprocessing import Process, JoinableQueue

import deepspeed  # 保留以兼容原环境
import torch
import pandas as pd
import numpy as np
import h5py
import cv2

# MM-related imports
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import MAP_FUNC
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          StopWordStoppingCriteria, PROMPT_TEMPLATE)
from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
from transformers import (AutoTokenizer, GenerationConfig, StoppingCriteriaList)

# Image processing imports
from PIL import Image, ImageFilter, ImageStat
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from torch.utils.data import Dataset, DataLoader

# 尝试导入 CONCH
try:
    from conch.open_clip_custom import create_model_from_pretrained
    CONCH_AVAILABLE = True
except ImportError:
    CONCH_AVAILABLE = False

Image.MAX_IMAGE_PIXELS = None
TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


# =============================================================================
# Part 1: WSI Patch Extraction & Feature Extraction Classes (Original Verified Logic)
# =============================================================================

class TileWorker(Process):
    """A child process that generates and writes tiles."""
    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds, quality, threshold):
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        self._queue = queue
        self._slidepath = slidepath
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._quality = quality
        self._threshold = threshold
        self._slide = None
    
    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break
            associated, level, address, outfile = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated
            try:
                tile = dz.get_tile(level, address)
                edge = tile.filter(ImageFilter.FIND_EDGES)
                edge = ImageStat.Stat(edge).sum
                edge = np.mean(edge) / (self._tile_size ** 2)
                w, h = tile.size
                if edge > self._threshold:
                    if not (w == self._tile_size and h == self._tile_size):
                        tile = tile.resize((self._tile_size, self._tile_size))
                    tile.save(outfile, quality=self._quality)
            except:
                pass
            self._queue.task_done()
    
    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)


class DeepZoomImageTiler:
    """Handles generation of tiles for a single image."""
    def __init__(self, dz, basename, target_levels, mag_base, img_format, associated, queue):
        self._dz = dz
        self._basename = basename
        self._format = img_format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._target_levels = target_levels
        self._mag_base = int(mag_base)
    
    def run(self):
        self._write_tiles()
    
    def _write_tiles(self):
        target_levels = [self._dz.level_count - i - 1 for i in self._target_levels]
        mag_list = [int(self._mag_base / 2 ** i) for i in self._target_levels]
        
        mag_idx = 0
        for level in range(self._dz.level_count):
            if not (level in target_levels):
                continue
            
            tiledir = os.path.join("%s_files" % self._basename, str(mag_list[mag_idx]))
            if not os.path.exists(tiledir):
                os.makedirs(tiledir)
            
            cols, rows = self._dz.level_tiles[level]
            
            for row in range(rows):
                for col in range(cols):
                    tilename = os.path.join(tiledir, '%d_%d.%s' % (col, row, self._format))
                    if not os.path.exists(tilename):
                        self._queue.put((self._associated, level, (col, row), tilename))
                    self._tile_done()
            
            mag_idx += 1
    
    def _tile_done(self):
        self._processed += 1


class DeepZoomStaticTiler:
    """Handles generation of tiles for all images in a slide."""
    def __init__(self, slidepath, basename, mag_levels, base_mag, objective, 
                 img_format, tile_size, overlap, limit_bounds, quality, 
                 workers, threshold):
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._format = img_format
        self._tile_size = tile_size
        self._overlap = overlap
        self._mag_levels = mag_levels
        self._base_mag = base_mag
        self._objective = objective
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        
        for _ in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                       limit_bounds, quality, threshold).start()
    
    def run(self):
        self._run_image()
        self._shutdown()
    
    def _run_image(self, associated=None):
        if associated is None:
            image = self._slide
            basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap,
                              limit_bounds=self._limit_bounds)
        
        MAG_BASE = self._slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
        if MAG_BASE is None:
            MAG_BASE = self._objective
        else:
            MAG_BASE = float(MAG_BASE)
        
        print(f"  Slide detected MAG: {MAG_BASE}X")
        
        first_level = int(math.log2(float(MAG_BASE) / self._base_mag))
        target_levels = [i + first_level for i in self._mag_levels]
        target_levels.reverse()
        
        tiler = DeepZoomImageTiler(dz, basename, target_levels, MAG_BASE, 
                                  self._format, associated, self._queue)
        tiler.run()
    
    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)
    
    def _shutdown(self):
        for _ in range(self._workers):
            self._queue.put(None)
        self._queue.join()


class PatchDataset(Dataset):
    def __init__(self, patch_files, preprocess):
        self.patch_files = patch_files
        self.preprocess = preprocess
    def __len__(self): return len(self.patch_files)
    def __getitem__(self, idx):
        file_path = self.patch_files[idx]
        try:
            img = cv2.imread(file_path)
            if img is None: raise ValueError
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.preprocess(img)
            return img, idx
        except:
            return torch.zeros((3, 224, 224)), idx

class WSIInferencePipeline:
    def __init__(self, conch_checkpoint, device='cuda', work_dir='./cache'):
        self.conch_checkpoint = conch_checkpoint
        self.device = device
        self.work_dir = work_dir
        self.feature_extractor = None
        self.is_initialized = False
        
        # WSI Settings
        self.patch_size = 224
        self.base_mag = 20.0
        self.workers = 4
        self.quality = 70
        self.threshold = 15
        
    def initialize_model(self):
        if self.is_initialized: return
        if not CONCH_AVAILABLE:
            raise RuntimeError("CONCH is required for WSI processing but not available.")
        
        print(f"Loading CONCH from {self.conch_checkpoint}...")
        model, preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path=self.conch_checkpoint
        )
        self.model = model.to(self.device, dtype=torch.float16).eval()
        self.preprocess = preprocess
        self.is_initialized = True
        print("CONCH loaded.")

    def process_slide(self, slide_path):
        self.initialize_model()
        
        slide_name = Path(slide_path).stem
        feature_save_dir = os.path.join(self.work_dir, "features")
        os.makedirs(feature_save_dir, exist_ok=True)
        feature_path = os.path.join(feature_save_dir, f"{slide_name}.h5")
        
        if os.path.exists(feature_path):
            print(f"Using cached features for {slide_name}")
            return feature_path
            
        print(f"Processing raw slide: {slide_name}")
        
        temp_dir = tempfile.mkdtemp(prefix=f"{slide_name}_patches_")
        try:
            basename = os.path.join(temp_dir, "slide")
            tiler = DeepZoomStaticTiler(
                slide_path, basename, [0], self.base_mag, self.base_mag, 
                'jpeg', self.patch_size, 0, True, self.quality, self.workers, self.threshold
            )
            tiler.run()
            
            patches = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.jpeg'):
                        patches.append(os.path.join(root, file))
            
            if not patches:
                raise ValueError(f"No patches extracted from {slide_path}")
                
            print(f"Extracted {len(patches)} patches.")
            features = self._extract_features_from_files(patches)
            
            with h5py.File(feature_path, 'w') as f:
                f.create_dataset('features', data=features)
            print(f"Features saved to {feature_path}")
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
                
        return feature_path

    def _extract_features_from_files(self, patch_files):
        dataset = PatchDataset(patch_files, self.preprocess)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False, 
                              num_workers=4, pin_memory=True)
        
        all_features = np.zeros((len(patch_files), 512), dtype=np.float32)
        
        with torch.no_grad():
            for batch_imgs, indices in dataloader:
                batch_imgs = batch_imgs.to(self.device, dtype=torch.float16)
                feats = self.model.encode_image(batch_imgs, proj_contrast=False, normalize=False)
                all_features[indices.numpy()] = feats.cpu().numpy()
        
        return all_features

# =============================================================================
# Part 2: Inference Setup
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Test model with WSI Support')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--checkpoint', default=None, help='LLM checkpoint file')
    parser.add_argument('--test_json', default=None, help='path to JSON input')
    parser.add_argument('--test_slide_csv', default=None, help='path to CSV input')
    parser.add_argument('--test_output_csv', default=None, help='output CSV path')
    parser.add_argument('--conch_checkpoint', default=None, help='Path to CONCH checkpoint.')
    parser.add_argument('--cache_dir', default='./wsi_cache', help='Directory to store extracted features')
    parser.add_argument('--torch-dtype', default='bf16', choices=TORCH_DTYPE_MAP.keys())
    parser.add_argument('--work-dir', help='directory to save metrics')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--merge-strategy', default='concat', choices=['concat', 'mean'])
    parser.add_argument('--sample_num', type=int, default=20480)
    parser.add_argument('--perturb_mode', default='none', choices=['none', 'mask', 'gaussian'])
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--patch_name_dir', default=None)
    parser.add_argument('--patch_file_pattern', default='{slide_name}.csv')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def register_function(cfg_dict):
    if isinstance(cfg_dict, dict):
        for key, value in dict.items(cfg_dict):
            if isinstance(value, FunctionType):
                value_str = str(value)
                if value_str not in MAP_FUNC:
                    MAP_FUNC.register_module(module=value, name=value_str)
                cfg_dict[key] = value_str
            else:
                register_function(value)
    elif isinstance(cfg_dict, (list, tuple)):
        for value in cfg_dict:
            register_function(value)

# =============================================================================
# Part 3: Data Loading
# =============================================================================

def load_feature_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.shape[1] > 513: df = df.iloc[:, 2:514]
    else: df = df.iloc[:, :512]
    return df

def load_feature_array(path: str) -> np.ndarray:
    if path.endswith('.csv'):
        df = load_feature_csv(path)
        arr = df.to_numpy()
    elif path.endswith(('.h5', '.hdf5', '.ibl.h5')):
        with h5py.File(path, 'r') as f:
            if 'features' not in f: raise KeyError(f"'features' missing in {path}")
            arr = f['features'][:]
    else:
        raise ValueError(f'Unsupported feature file type: {path}')
    if arr.ndim != 2: arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] != 512: arr = arr[:, :512]
    return arr.astype(np.float32)

def resolve_data_path(path: str, wsi_pipeline: WSIInferencePipeline = None) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if not path.endswith(('.csv', '.h5', '.hdf5', '.ibl.h5')):
        if wsi_pipeline is None:
            raise ValueError(f"Input is a raw slide ({path}) but CONCH/WSI pipeline is not initialized.")
        return wsi_pipeline.process_slide(path)
    else:
        return path

def build_image_tensor(case_names: List[str], wsi_pipeline: WSIInferencePipeline = None,
                       sample_num: int = 10240, merge_strategy: str = 'concat',
                       perturb_mode: str = 'none', noise_std: float = 0.1,
                       patch_names: List[str] = None) -> torch.Tensor:
    arrays = []
    for p in case_names:
        if not isinstance(p, str): continue
        final_path = resolve_data_path(p, wsi_pipeline)
        arr = load_feature_array(final_path)
        arrays.append(arr)
    if not arrays: raise ValueError("No feature arrays loaded.")
    if merge_strategy == 'concat':
        image_array = np.concatenate(arrays, axis=0)
    else:
        pooled = [arr.mean(axis=0, keepdims=True) for arr in arrays]
        image_array = np.concatenate(pooled, axis=0)
    if sample_num is not None and image_array.shape[0] > sample_num:
        indices = np.linspace(0, image_array.shape[0] - 1, sample_num, dtype=int)
        image_array = image_array[indices]
    image = torch.from_numpy(image_array).half().cuda()
    return image.unsqueeze(0) 

def iter_cases_from_csv(csv_path: str):
    import ast
    df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='replace')
    if 'Slide' not in df.columns: df['Slide'] = None
    for i in range(df.shape[0]):
        row = df.loc[i]
        slide_raw = row.get('Slide')
        case_names = []
        if isinstance(slide_raw, list): case_names = slide_raw
        elif isinstance(slide_raw, str):
            slide_raw = slide_raw.strip()
            if slide_raw.startswith('[') and slide_raw.endswith(']'):
                try: case_names = ast.literal_eval(slide_raw)
                except: case_names = [slide_raw]
            else: case_names = [slide_raw]
        final_case_names = [cn for cn in case_names if cn]
        yield {"index": i, "meta": row.to_dict(), "question": row.get('Question', ''), "case_names": final_case_names}

def iter_cases_from_json(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    for i, item in enumerate(data):
        cn = item.get('case_name', [])
        if isinstance(cn, str): cn = [cn]
        yield {"index": i, "meta": item, "question": item.get('Question', ''), "case_names": cn}

# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    
    if not osp.isfile(args.config):
        try: args.config = cfgs_name_path[args.config]
        except KeyError: raise FileNotFoundError(f'Cannot find {args.config}')
    
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options: cfg.merge_from_dict(args.cfg_options)
    register_function(cfg._cfg_dict)
    
    if args.work_dir: cfg.work_dir = args.work_dir
    else: cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    if 'runner_type' not in cfg: runner = Runner.from_cfg(cfg)
    else: runner = RUNNERS.build(cfg)

    state_dict = guess_load_checkpoint(args.checkpoint) if args.checkpoint else None
    if state_dict:
        runner.model.load_state_dict(state_dict, strict=False)
        print(f'Load checkpoint from {args.checkpoint}')
    runner.model.eval()

    llm_name_or_path = 'Qwen/Qwen2.5-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(llm_name_or_path, device_map="auto", trust_remote_code=True, encode_special_tokens=True)
    llm = runner.model.llm.eval().half().cuda()
    LongNet_encoder = runner.model.LongNet_encoder.to(torch.float16).half().cuda().eval()
    projector = runner.model.projector.to(torch.float16).half().cuda().eval()
    
    wsi_pipeline = None
    if args.conch_checkpoint:
        wsi_pipeline = WSIInferencePipeline(conch_checkpoint=args.conch_checkpoint, work_dir=args.cache_dir)
    
    columns = ['ID', 'Slide', 'Question', 'A', 'B', 'C', 'D', 'Answer', 'Output']
    df_output = pd.DataFrame(columns=columns)
    
    if args.test_json: iter_data = iter_cases_from_json(args.test_json)
    elif args.test_slide_csv: iter_data = iter_cases_from_csv(args.test_slide_csv)
    else: raise ValueError("Provide --test_json or --test_slide_csv")

    for case in iter_data:
        idx = case['index']
        meta = case['meta']
        question = case['question']
        case_names = case['case_names'] 
        
        print(f"\n[{idx}] Processing: {case_names}")
        
        try:
            image_tensor = build_image_tensor(
                case_names, wsi_pipeline=wsi_pipeline, sample_num=args.sample_num,
                merge_strategy=args.merge_strategy, perturb_mode=args.perturb_mode, noise_std=args.noise_std
            )
        except Exception as e:
            print(f"Error loading data for case {idx}: {e}")
            import traceback; traceback.print_exc()
            continue

        prompt_template = PROMPT_TEMPLATE.qwen_chat
        options = []
        for opt in ['A', 'B', 'C', 'D']:
            val = meta.get(opt)
            if val and isinstance(val, str) and val.strip():
                options.append(f"{opt}. {val}")
        options_str = '\n'.join(options)
        
        content = f"{question}\n{options_str}" if options_str else question
        print("Question: ", content)
        sys_prompt = ''
        instruction = prompt_template.get('INSTRUCTION', '{input}')
        model_input = DEFAULT_IMAGE_TOKEN + '\n' + content
        final_prompt = (sys_prompt + instruction).format(input=model_input, round=1, **runner.cfg)

        chunk_encode = []
        for i, chunk in enumerate(final_prompt.split(DEFAULT_IMAGE_TOKEN)):
            if i == 0: cur = tokenizer.encode(chunk)
            else: cur = tokenizer.encode(chunk, add_special_tokens=False)
            chunk_encode.append(cur)
        
        input_ids = []
        for i, cur in enumerate(chunk_encode):
            input_ids.extend(cur)
            if i != len(chunk_encode) - 1: input_ids.append(IMAGE_TOKEN_INDEX)
        
        input_ids = torch.tensor(input_ids).cuda().unsqueeze(0)

        with torch.no_grad():
            img_embeds = image_tensor.to(llm.dtype)
            img_enc_out = LongNet_encoder(
                src_tokens=None, token_embeddings=img_embeds.permute(1, 0, 2)
            )["encoder_out"]
            
            img_enc_out = img_enc_out.permute(1, 0, 2)
            pixel_values = projector(img_enc_out)

            mm_inputs = prepare_inputs_labels_for_multimodal(
                llm=llm, input_ids=input_ids, pixel_values=pixel_values.half()
            )

            gen_config = GenerationConfig(
                max_new_tokens=args.max_tokens,
                do_sample=False, 
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                output_attentions=True,          
                return_dict_in_generate=True, 
            )
            
            stop_criteria = StoppingCriteriaList()
            stop_words = prompt_template.get('STOP_WORDS', [])
            if stop_words: stop_criteria.append(StopWordStoppingCriteria(tokenizer, stop_words))


            generate_output = llm.generate(
                **mm_inputs,
                generation_config=gen_config,
                bos_token_id=tokenizer.bos_token_id,
                stopping_criteria=stop_criteria,
                output_attentions=False,
                output_hidden_states=False,
                return_dict_in_generate=True,  
                use_cache=False,
            )
            

            sequences = generate_output.sequences
            response = tokenizer.decode(sequences[0])
            if response.endswith('<|im_end|>'): response = response[:-10]
            
            print(f"Output: {response}")

        meta['Output'] = response

        row_save = {
            'ID': meta.get('ID', idx + 1),
            'Slide': ', '.join(case_names) if case_names else '',
            'Question': question,
            'Answer': meta.get('Answer', ''),
            'Output': response
        }

        for opt in ['A', 'B', 'C', 'D']:
            row_save[opt] = meta.get(opt, '')
            
        df_output.loc[len(df_output)] = row_save
        
        if args.test_output_csv:
            df_output.to_csv(args.test_output_csv, index=False)

    print("Inference finished.")

if __name__ == '__main__':
    main()