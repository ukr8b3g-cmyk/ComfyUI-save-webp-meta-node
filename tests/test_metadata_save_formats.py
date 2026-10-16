"""Real Pillow round trips; these do not certify browser/GPU generation."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import piexif
import piexif.helper
from PIL import Image, features
from test_webp_save import webp_save, folder_paths
from test_metadata_resolver import base, n

class Tensor:
    def __init__(self, channels=4):
        self.array=np.linspace(0,1,13*17*channels,dtype=np.float32).reshape(13,17,channels)
        if channels==4:self.array[0,:,3]=0
        self.shape=self.array.shape
    def detach(self):return self
    def cpu(self):return self
    def numpy(self):return self.array

class SaveFormatTests(unittest.TestCase):
    def setUp(self):self.saver=webp_save.SaveWebPMeta()
    def run_save(self, directory, p, images, fmt='png', extra=None):
        folder_paths.get_output_directory=lambda:directory
        return self.saver.save_webp(images,filename_prefix='%model%_%seed%_%width%x%height%',file_format=fmt,quality=70,id='save',prompt=p,
                                    extra_pnginfo=extra if extra is not None else {'workflow':{'nodes':[], 'links':[]}})
    def read_meta(self, path):
        with Image.open(path) as im:
            if im.format=='PNG':return im.info['parameters'],json.loads(im.info['prompt']),json.loads(im.info['workflow'])
            exif=piexif.load(im.info['exif'])
            params=piexif.helper.UserComment.load(exif['Exif'][piexif.ExifIFD.UserComment])
            tags=[exif['0th'].get(k,b'').decode('utf-8') for k in webp_save.EXIF_TEXT_TAGS]
            return params,json.loads(next(v[7:] for v in tags if v.startswith('prompt:'))),json.loads(next(v[9:] for v in tags if v.startswith('workflow:')))

    def check_format(self, fmt):
        if fmt=='avif' and not features.check('avif'):self.skipTest('Pillow AVIF not available')
        p=base('SamplerCustomAdvanced');wf={'nodes':[], 'links':[], 'extra':{'日本語':'保持'}};before=copy.deepcopy((p,wf));image=Tensor()
        with tempfile.TemporaryDirectory() as d:
            result=self.run_save(d,p,[image],fmt,{'workflow':wf});f=result['ui']['images'][0]['filename'];path=Path(d)/f
            params,rp,rw=self.read_meta(path)
            self.assertEqual((rp,rw),before);self.assertEqual((p,wf),before)
            self.assertIn('Size: 17x13',params);self.assertIn('Seed: 18446744073709551615',params)
            self.assertIn('17x13',f);self.assertNotIn('CFG scale:',params);self.assertNotIn('Negative prompt:',params)
            self.assertNotIn('UNRELATED',params)
            with Image.open(path) as im:
                self.assertEqual(im.size,(17,13))
                if fmt in ('png','webp_lossless'):
                    np.testing.assert_array_equal(np.asarray(im.convert('RGBA')),(image.array*255).clip(0,255).astype('uint8'))
    def test_png(self):self.check_format('png')
    def test_webp(self):self.check_format('webp')
    def test_webp_lossless_rgba_exact(self):self.check_format('webp_lossless')
    def test_jpg(self):self.check_format('jpg')
    def test_avif(self):self.check_format('avif')

    def test_actual_tensor_geometry_after_resize(self):
        p=base();p['resize']=n('ImageScale',image=['decode',0],width=999,height=888,upscale_method='nearest-exact',crop='disabled')
        p['save']['inputs']['images']=['resize',0]
        with tempfile.TemporaryDirectory() as d:
            f=self.run_save(d,p,[Tensor()])['ui']['images'][0]['filename']
            params,*_=self.read_meta(Path(d)/f)
            self.assertIn('Size: 17x13',params);self.assertNotIn('1024x768',params);self.assertNotIn('999x888',params)

    def test_external_and_unsupported_images_still_save(self):
        for kind in ('LoadImage','SomeUnknownGenerator'):
            p=base();p['external']=n(kind,image='foo.png',seed=999,prompt='do not trust');p['save']['inputs']['images']=['external',0]
            with tempfile.TemporaryDirectory() as d:
                f=self.run_save(d,p,[Tensor()])['ui']['images'][0]['filename'];params,rp,_=self.read_meta(Path(d)/f)
                self.assertEqual(params,'Size: 17x13');self.assertEqual(rp,p)

    def test_exif_capacity_does_not_drop_workflow(self):
        p=base();extra={'workflow':{},**{f'other{i}':i for i in range(7)}}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'text-tag slots'):self.run_save(d,p,[Tensor()],'webp',extra)
            self.assertEqual(list(Path(d).iterdir()),[])
            self.assertEqual(len(self.run_save(d,p,[Tensor()],'png',extra)['ui']['images']),1)

    def test_large_jpeg_metadata_and_collision_fail_before_write(self):
        p=base();p['positive']['inputs']['text']='文字'*40000
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'APP1'):self.run_save(d,p,[Tensor()],'jpg')
            self.assertEqual(list(Path(d).iterdir()),[])
            with self.assertRaisesRegex(ValueError,'collision'):self.run_save(d,base(),[Tensor()],'png',{'parameters':'bad'})
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_batch_not_seed_plus_index_and_unique_names(self):
        p=base();p['latent']['inputs']['batch_size']=3
        with tempfile.TemporaryDirectory() as d:
            r=self.run_save(d,p,[Tensor(3),Tensor(3),Tensor(3)])
            names=[x['filename'] for x in r['ui']['images']];self.assertEqual(len(set(names)),3)
            for f in names:self.assertIn('Seed: 42',self.read_meta(Path(d)/f)[0])

    def test_mixed_producer_batch_gets_per_image_parameters(self):
        p=base();p['other']=copy.deepcopy(p['sampler']);p['other']['inputs']['seed']=71
        p['d2']=n('VAEDecode',samples=['other',0],vae=['vae',0]);p['batch']=n('ImageBatch',image1=['decode',0],image2=['d2',0]);p['save']['inputs']['images']=['batch',0]
        with tempfile.TemporaryDirectory() as d:
            r=self.run_save(d,p,[Tensor(),Tensor()]);params=[self.read_meta(Path(d)/x['filename'])[0] for x in r['ui']['images']]
            self.assertIn('Seed: 42',params[0]);self.assertIn('Seed: 71',params[1])

    def test_unresolved_lora_never_hashes_model_file(self):
        p=base();p['lora']=n('LoraLoaderModelOnly',model=['model',0],lora_name='absent.safetensors',strength_model=0);p['sampler']['inputs']['model']=['lora',0]
        with tempfile.TemporaryDirectory() as d, patch.object(webp_save,'_lora_file_hash_by_name',side_effect=AssertionError('weight read')), patch.object(webp_save.os,'walk',side_effect=AssertionError('recursive scan')):
            f=self.run_save(d,p,[Tensor()])['ui']['images'][0]['filename'];self.assertNotIn('<lora:',self.read_meta(Path(d)/f)[0])

    def test_public_interface_unchanged_and_package_import(self):
        types=self.saver.INPUT_TYPES()
        self.assertEqual(list(types['required']),['images','filename_prefix','file_format','quality'])
        self.assertEqual(types['hidden'],{'id':'UNIQUE_ID','prompt':'PROMPT','extra_pnginfo':'EXTRA_PNGINFO'})
        self.assertEqual(types['required']['quality'][1]['default'],70)
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('save_webp_package_test',root/'__init__.py',submodule_search_locations=[str(root)])
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        self.assertIn('SaveWebPMeta',module.NODE_CLASS_MAPPINGS)

if __name__=='__main__':unittest.main()
