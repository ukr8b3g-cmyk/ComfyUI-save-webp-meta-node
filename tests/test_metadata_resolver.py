"""Metadata contracts, using graphs rather than model inference.

Provider parity tests use pinned PUBLIC upstream source (set
SAVE_META_PROVIDER_SOURCES). No models, GPU or services are loaded.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from metadata_resolver import resolve_metadata, to_legacy_info
from metadata_resolver.graph import Graph, reference
from metadata_resolver.providers import VerifiedProviders
from metadata_resolver.resolver import Resolver
from metadata_resolver import profiles as P


def n(kind, **inputs):
    return {'class_type': kind, 'inputs': inputs}


def base(kind='KSampler'):
    p = {
        'old': n('KSampler', seed=999, steps=99, positive=['oldtext', 0], model=['oldmodel', 0]),
        'oldtext': n('CLIPTextEncode', text='UNRELATED'),
        'oldmodel': n('UNETLoader', unet_name='unrelated.safetensors'),
        'model': n('UNETLoader', unet_name='renamed.safetensors', weight_dtype='default'),
        'clip': n('CLIPLoader', clip_name='encoder.safetensors', type='krea2'),
        'vae': n('VAELoader', vae_name='video_vae.safetensors'),
        'positive': n('CLIPTextEncode', text='actual positive', clip=['clip', 0]),
        'negative': n('CLIPTextEncode', text='', clip=['clip', 0]),
        'latent': n('EmptyLatentImage', width=1024, height=768, batch_size=1),
        'sampler': n(kind, seed=42, noise_seed=43, steps=20, cfg=1, denoise=1, sampler_name='euler', scheduler='simple',
                     positive=['positive', 0], negative=['negative', 0], model=['model', 0], latent_image=['latent', 0]),
        'decode': n('VAEDecode', samples=['sampler', 0], vae=['vae', 0]),
        'save': n('SaveWebPMeta', images=['decode', 0]),
    }
    if kind.startswith('SamplerCustom'):
        p.update(noise=n('RandomNoise', noise_seed=2**64-1),
                 guider=n('BasicGuider', model=['model', 0], conditioning=['positive', 0]),
                 algorithm=n('KSamplerSelect', sampler_name='res_multistep'),
                 schedule=n('BasicScheduler', scheduler='simple', steps=23, denoise=1))
        p['sampler']['inputs'].update(noise=['noise', 0], guider=['guider', 0], sampler=['algorithm', 0], sigmas=['schedule', 0])
    return p


def report(p, **kw):
    return resolve_metadata(prompt=p, save_id='save', **kw)


def info(p, **kw):
    return to_legacy_info(report(p, **kw))


class GraphTests(unittest.TestCase):
    def test_active_sampler_empty_negative_and_input_immutability(self):
        p = base(); before = copy.deepcopy(p)
        got = info(p)
        self.assertEqual((got['seed'], got['prompt'], got['negative_prompt']), (42, 'actual positive', ''))
        self.assertEqual(p, before)
        self.assertEqual(got['model'], 'renamed.safetensors')
        self.assertEqual(got['vae'], 'video_vae.safetensors')

    def test_unique_save_and_no_global_fallback(self):
        p = base(); self.assertEqual(to_legacy_info(resolve_metadata(prompt=p))['seed'], 42)
        del p['save']; self.assertNotIn('seed', to_legacy_info(resolve_metadata(prompt=p)))
        p = base(); p['save2'] = n('SaveWebPMeta', images=['decode', 0])
        self.assertNotIn('seed', to_legacy_info(resolve_metadata(prompt=p)))
        self.assertNotIn('seed', to_legacy_info(resolve_metadata(prompt=p, save_id='missing')))

    def test_external_image_never_uses_nearby_sampler(self):
        p = base(); p['load'] = n('LoadImage', image='input.png'); p['save']['inputs']['images'] = ['load', 0]
        self.assertEqual(report(p).status, 'external'); self.assertNotIn('model', info(p))

    def test_unknown_image_transform_does_not_use_its_other_inputs(self):
        p = base(); p['unknown'] = n('UnverifiedMagicImage', image=['decode', 0], positive=['positive', 0]); p['save']['inputs']['images'] = ['unknown', 0]
        self.assertNotIn('seed', info(p)); self.assertEqual(report(p).status, 'partial')

    def test_switch_boolean_literal_and_link(self):
        for selector, expected in [(True, 'actual positive'), (False, 'UNRELATED'), (['bool', 0], 'actual positive')]:
            with self.subTest(selector=selector):
                p = base(); p['bool'] = n('PrimitiveBoolean', value=True)
                p['s'] = n('ComfySwitchNode', switch=selector, on_true=['positive', 0], on_false=['oldtext', 0])
                p['sampler']['inputs']['positive'] = ['s', 0]
                self.assertEqual(info(p)['prompt'], expected)
        for selector in [0, 1, 'true', ['unknown', 0], None, ['bool', True]]:
            p['s']['inputs']['switch'] = selector
            self.assertNotIn('prompt', info(p))

    def test_reference_validation_and_cycle(self):
        for x in [['x', True], ['x', -1], ['x', 0, 1], [False, 0], ['x', 0.0]]:
            self.assertIsNone(reference(x))
        p = base(); p['loop'] = n('StringConcatenate', string_a=['loop', 0], string_b='b', delimiter='')
        p['positive']['inputs']['text'] = ['loop', 0]
        self.assertNotIn('prompt', info(p)); self.assertEqual(info(p)['seed'], 42)
        p['decode']['inputs']['samples'] = ['sampler', True]
        self.assertNotIn('seed', info(p))

    def test_seed_precision_invalid_boolean_and_nonfinite(self):
        for seed in [0, 2**53+1, 2**64-1]:
            p = base(); p['sampler']['inputs']['seed'] = seed
            self.assertEqual(info(p)['seed'], seed)
        for seed in [True, -1, 2**64, float('nan'), '42', 1.0]:
            p['sampler']['inputs']['seed'] = seed; self.assertNotIn('seed', info(p))
        p['sampler']['inputs']['cfg'] = float('inf'); self.assertNotIn('cfg', info(p))

    def test_api_does_not_fill_holes_with_old_workflow(self):
        wf = {'nodes': [{'id':'save','type':'SaveWebPMeta','inputs':[{'name':'images','link':1}]},
                        {'id':'sampler','type':'KSampler','widgets_values':[999,'fixed',33,7,'euler','normal',1]}],
              'links': [[1, 'sampler',0,'save',0,'IMAGE']]}
        self.assertNotIn('seed', to_legacy_info(resolve_metadata(prompt={}, workflow=wf, save_id='save')))
        p = base(); p['sampler']['inputs']['seed'] = ['unsupported', 0]
        self.assertNotIn('seed', to_legacy_info(resolve_metadata(prompt=p, workflow=wf, save_id='save')))

    def test_ui_named_priority_and_advanced_widget_order(self):
        wf = {'nodes': [{'id':1,'type':'KSamplerAdvanced','widgets_values':['enable',43,'fixed',20,7,'euler','simple',3,18,'disable'],
                         'widgets_values_named':{'noise_seed':99}},
                        {'id':2,'type':'VAEDecode','inputs':[{'name':'samples','link':1}]},
                        {'id':3,'type':'SaveWebPMeta','inputs':[{'name':'images','link':2}]}],
              'links':[[1,1,0,2,0,'LATENT'], [2,2,0,3,0,'IMAGE']]}
        r = resolve_metadata(workflow=wf, save_id=3); got = to_legacy_info(r)
        self.assertEqual((got['seed'], got['steps'], got['cfg']), (99,20,7)); self.assertNotIn('denoise', got)
        self.assertEqual(r.fields['start_at_step'].value, 3)
        wf['nodes'][0]['inputs'] = [{'name':'noise_seed','link':777}]
        self.assertNotIn('seed', to_legacy_info(resolve_metadata(workflow=wf, save_id=3)))

    def test_ui_subgraph_instances_keep_namespace(self):
        definition = {'id':'sg','inputNode':{'id':-10}, 'outputNode':{'id':-20},
                      'inputs':[{'name':'noise_seed'}], 'outputs':[{'name':'image','linkIds':[3]}],
                      'nodes':[{'id':1,'type':'KSampler','widgets_values':[0,'fixed',20,1,'euler','simple',1],
                                'inputs':[{'name':'seed','link':1}]},
                               {'id':2,'type':'VAEDecode','inputs':[{'name':'samples','link':2}]}],
                      'links':[{'id':1,'origin_id':-10,'origin_slot':0,'target_id':1,'target_slot':0},
                               {'id':2,'origin_id':1,'origin_slot':0,'target_id':2,'target_slot':0},
                               {'id':3,'origin_id':2,'origin_slot':0,'target_id':-20,'target_slot':0}]}
        wf = {'definitions':{'subgraphs':[definition]}, 'nodes':[{'id':18,'type':'sg','widgets_values_named':{'noise_seed':21}},
              {'id':19,'type':'sg','widgets_values_named':{'noise_seed':22}},
              {'id':20,'type':'SaveWebPMeta','inputs':[{'name':'images','link':5}]}], 'links':[[5,19,0,20,0,'IMAGE']]}
        before = copy.deepcopy(wf); r = resolve_metadata(workflow=wf,save_id=20)
        self.assertEqual(to_legacy_info(r)['seed'],22); self.assertEqual(r.producers[-1]['node_id'],'19:1'); self.assertEqual(before,wf)

    def test_opaque_id_and_duplicate_definition_safety(self):
        p=base();p['18:5']=n('CLIPTextEncode',text='namespaced',clip=['clip',0]);p['5']=n('CLIPTextEncode',text='root',clip=['clip',0])
        p['sampler']['inputs']['positive']=['18:5',0];self.assertEqual(info(p)['prompt'],'namespaced')
        r=resolve_metadata(workflow={'nodes':[], 'definitions':{'subgraphs':[{'id':'same'},{'id':'same'}]}})
        self.assertEqual(r.status,'partial');self.assertTrue(r.warnings)

    def test_bad_objects_not_executed(self):
        class Evil(dict):
            def get(self,*args):raise AssertionError('getter should not execute')
        p=base();p['bad']=Evil();p['save']['inputs']['images']=['bad',0]
        self.assertNotIn('seed',info(p))
        p={str(i):n('Dummy') for i in range(10001)}
        self.assertEqual(resolve_metadata(prompt=p).warnings[0]['code'],'metadata_analysis_failed')

    def test_limits_terminate_without_execution(self):
        p=base()
        for i in range(260): p[f'x{i}']=n('StringConcatenate', string_a=[f'x{i+1}',0],string_b='',delimiter='')
        p['positive']['inputs']['text']=['x0',0]
        self.assertNotIn('prompt',info(p)); self.assertEqual(info(p)['seed'],42)
        p['positive']['inputs']['text']='x'*(1024*1024+1)
        self.assertNotIn('prompt',info(p))


class SamplingTests(unittest.TestCase):
    def test_custom_advanced_h3_with_frame(self):
        p=base('SamplerCustomAdvanced')
        p['positive']=n('MiniMaxH3ReferenceToVideo', prompt='H3 literal', clip=['clip',0],vae=['vae',0],width=1280,height=768,length=5,
                         **{'ref_images.ref_image_0':['olddecode',0]})
        p['sampler']['inputs']['latent_image']=['positive',1]
        p['frame']=n('ImageFromBatch',image=['decode',0],batch_index=0,length=1);p['save']['inputs']['images']=['frame',0]
        r=report(p,image={'width':2048,'height':930,'channels':4,'batch_index':0},image_index=0); v=to_legacy_info(r)
        self.assertEqual((v['seed'],v['steps'],v['sampler'],v['prompt'],v['model_family']), (2**64-1,23,'res_multistep','H3 literal','MiniMax H3'))
        self.assertNotIn('cfg',v);self.assertNotIn('negative_prompt',v);self.assertEqual(v['width'],2048)
        self.assertEqual(r.fields['frame_index'].value,0);self.assertEqual(r.fields['h3_length_requested'].value,5)
        json.dumps(r.as_dict(),allow_nan=False)

    def test_custom_cfg_dual_and_disabled_noise(self):
        p=base('SamplerCustomAdvanced');p['guider']=n('CFGGuider',model=['model',0],positive=['positive',0],negative=['negative',0],cfg=2.5)
        self.assertEqual(info(p)['cfg'],2.5)
        p['guider']=n('DualCFGGuider',model=['model',0],cond1=['positive',0],cond2=['negative',0],negative=['oldtext',0],cfg_conds=2,cfg_cond2_negative=3,style='nested')
        self.assertNotIn('cfg',info(p));self.assertEqual(report(p).fields['cfg_conds'].value,2)
        p['noise']=n('DisableNoise');self.assertNotIn('seed',info(p))
        p['guider']=n('DualModelGuider',model=['model',0],model_negative=['oldmodel',0],positive=['positive',0],cfg=4)
        self.assertNotIn('negative_prompt',info(p));self.assertEqual(report(p).fields['model_negative'].value,['unrelated.safetensors'])

    def test_custom_direct_and_denoised_output(self):
        p=base('SamplerCustom');p['decode']['inputs']['samples']=['sampler',1]
        self.assertEqual((info(p)['seed'],info(p)['steps']),(43,23))
        self.assertEqual(report(p).producers[-1]['output_variant'],'denoised_output')
        p['decode']['inputs']['samples']=['sampler',2];self.assertNotIn('seed',info(p))

    def test_unknown_sigmas_does_not_borrow_steps(self):
        p=base('SamplerCustomAdvanced');p['split']=n('UnverifiedSigmas',sigmas=['schedule',0],steps=3)
        p['sampler']['inputs']['sigmas']=['split',0]
        self.assertNotIn('steps',info(p));self.assertNotIn('scheduler',info(p));self.assertIn('seed',info(p))
        p['sampler']['inputs']['sigmas']=['schedule',0];p['schedule']['inputs']['denoise']=0
        self.assertEqual(info(p)['steps'],0);self.assertEqual(report(p).fields['steps_requested'].value,23)

    def test_scheduler_contracts_and_noise_switch(self):
        for kind,expected in P.SCHEDULES.items():
            p=base('SamplerCustomAdvanced');p['schedule']=n(kind,steps=17,denoise=1)
            self.assertEqual(info(p)['scheduler'],expected);self.assertEqual(info(p)['steps'],17)
        p['nswitch']=n('ComfySwitchNode',switch=True,on_true=['noise',0],on_false=['missing',0]);p['sampler']['inputs']['noise']=['nswitch',0]
        self.assertEqual(info(p)['seed'],2**64-1)
        p['schedule']['inputs']['steps']=17.5;self.assertNotIn('steps',info(p))

    def test_h3_guides_shift_and_latent_is_not_negative(self):
        p=base('SamplerCustomAdvanced');p['positive']=n('MiniMaxH3ImageToVideo',prompt='H3 prompt',clip=['clip',0],width=1280,height=768,length=5)
        p['guide']=n('MiniMaxH3AddGuide',positive=['positive',0],image=['unrelated_image',0],frame_idx=0)
        p['shift']=n('MiniMaxH3SigmaShift',model=['model',0],shift_video=12,shift_audio=3)
        p['guider']['inputs'].update(conditioning=['guide',0],model=['shift',0])
        r=report(p);self.assertEqual(to_legacy_info(r)['prompt'],'H3 prompt');self.assertEqual(to_legacy_info(r)['model_family'],'MiniMax H3')
        self.assertEqual(r.fields['model_modifiers'].value[-1]['settings']['shift_video']['value'],12)
        p['guider']=n('CFGGuider',positive=['positive',0],negative=['positive',1],model=['model',0],cfg=1)
        self.assertNotIn('negative_prompt',info(p))

    def test_multistage_does_not_mix(self):
        p=base();p['base']=copy.deepcopy(p['sampler']);p['base']['inputs']['seed']=51
        p['resize']=n('LatentUpscale',samples=['base',0],width=2048,height=1536)
        p['sampler']['inputs']['latent_image']=['resize',0]
        r=report(p);self.assertEqual(to_legacy_info(r)['seed'],42)
        self.assertEqual([x['node_id'] for x in r.producers],['base','sampler'])
        self.assertEqual(r.fields['ancestor_stages'].value[0]['seed']['value'],51)

    def test_batch_index_producer_selection_and_ambiguity(self):
        p=base();p['other']=copy.deepcopy(p['sampler']);p['other']['inputs']['seed']=77
        p['otherdecode']=n('VAEDecode',samples=['other',0],vae=['vae',0])
        p['batch']=n('ImageBatch',image1=['decode',0],image2=['otherdecode',0]);p['save']['inputs']['images']=['batch',0]
        self.assertEqual(info(p,image_index=0)['seed'],42);self.assertEqual(info(p,image_index=1)['seed'],77)
        self.assertNotIn('seed',info(p));self.assertEqual(report(p).status,'ambiguous')
        p['frame']=n('ImageFromBatch',image=['batch',0],batch_index=-1,length=1);p['save']['inputs']['images']=['frame',0]
        self.assertEqual(info(p,image_index=0)['seed'],77)


class ConditioningTests(unittest.TestCase):
    def test_every_explicit_encoder(self):
        for kind, keys in P.TEXT_ENCODERS.items():
            with self.subTest(kind=kind):
                p=base();p['positive']=n(kind,clip=['clip',0],**{k:'same prompt' for k in keys})
                self.assertEqual(info(p)['prompt'],'same prompt')
        for kind in P.DUAL_ENCODERS:
            p=base();p['positive']=n(kind,clip=['clip',0],prompt='positive',negative_prompt='negative')
            # Slot defines the text, not the downstream role.
            p['sampler']['inputs'].update(positive=['positive',1],negative=['positive',0])
            self.assertEqual((info(p)['prompt'],info(p)['negative_prompt']),('negative','positive'))
            p['sampler']['inputs']['positive']=['positive',2];self.assertNotIn('prompt',info(p))

    def test_multiencoder_fragments_and_loader_counts(self):
        p=base();p['positive']=n('CLIPTextEncodeHiDream',clip=['clip',0],clip_l='L',clip_g='G',t5xxl='T',llama='M')
        p['clip']=n('QuadrupleCLIPLoader',clip_name1='l',clip_name2='g',clip_name3='t',clip_name4='m')
        r=report(p);self.assertEqual(r.fields['text_encoders'].value,['l','g','t','m'])
        self.assertEqual(info(p)['prompt'],'[clip_l] L\n[clip_g] G\n[t5xxl] T\n[llama] M')
        for kind, keys in P.CLIP_LOADERS.items():
            p['clip']=n(kind,**{k:k+'.safetensors' for k in keys}); self.assertEqual(len(report(p).fields['text_encoders'].value),len(keys))

    def test_guidance_zeroout_and_reference_order(self):
        p=base();p['positive']=n('TextEncodeQwenImage21',prompt='keep',negative_prompt='',clip=['clip',0],
                                  **{'images.image_10':['ten',0],'images.image_2':['two',0]})
        p['g']=n('FluxGuidance',conditioning=['positive',0],guidance=3.5);p['sampler']['inputs']['positive']=['g',0]
        self.assertEqual(info(p)['cfg'],1);self.assertEqual(report(p).fields['embedded_guidance'].value,3.5)
        refs=report(p).fields['positive_references'].value;self.assertEqual([x['input'] for x in refs],['images.image_2','images.image_10'])
        p['g']=n('ConditioningZeroOut',conditioning=['positive',0]);self.assertEqual(info(p)['prompt'],'')

    def test_dual_conditioning_passthrough_slot(self):
        for kind in P.DUAL_COND:
            p=base();p['wrapper']=n(kind,positive=['positive',0],negative=['oldtext',0])
            p['sampler']['inputs'].update(positive=['wrapper',1],negative=['wrapper',0])
            self.assertEqual((info(p)['prompt'],info(p)['negative_prompt']),('UNRELATED','actual positive'))

    def test_native_clip_stack_and_missing_strength(self):
        p=base();p['stack']=n('LoadLoraTextEncoder',clip=['clip',0],**{'loras.19.lora_name':'clip.safetensors','loras.19.enabled':True,'loras.19.strength':-.75})
        p['positive']['inputs']['clip']=['stack',0];r=report(p)
        self.assertEqual(r.fields['lora_entries'].value[0]['target'],'clip');self.assertNotIn('loras',to_legacy_info(r))
        del p['stack']['inputs']['loras.19.strength'];r=report(p);self.assertEqual(r.status,'partial')
        self.assertTrue(any(w['code']=='unresolved_lora_rows' for w in r.warnings))

    def test_unknown_part_not_published_as_complete(self):
        p=base();p['part']=n('StringConcatenate',string_a='known',string_b=['LLM',0],delimiter=' ')
        p['positive']['inputs']['text']=['part',0];self.assertNotIn('prompt',info(p))
        p['part']['inputs']['string_b']='';self.assertEqual(info(p)['prompt'],'known ')
        p['part']['inputs']['delimiter']=['unknown',0];self.assertNotIn('prompt',info(p))
        p['combo']=n('ConditioningAverage',conditioning_to=['positive',0],conditioning_from=['oldtext',0],conditioning_to_strength=.5)
        p['sampler']['inputs']['positive']=['combo',0];self.assertNotIn('prompt',info(p))

    def test_model_family_only_from_evidence_and_rng_not_clip_device(self):
        p=base();p['clip']['inputs'].update(type='stable_diffusion',device='cpu');p['model']['inputs']['unet_name']='qwen_image_2.1.safetensors'
        self.assertNotIn('model_family',info(p));self.assertNotIn('rng_source',info(p))
        for typ,family in P.CLIP_FAMILY.items():
            p['clip']['inputs']['type']=typ;self.assertEqual(info(p)['model_family'],family)

    def test_native_lora_order_disabled_negative_and_precision(self):
        p=base();p['stack']=n('LoadLoraModel',model=['model',0],**{
            'loras.10.lora_name':'repeat.safetensors','loras.10.strength':-.001,'loras.10.enabled':True,
            'loras.2.lora_name':'repeat.safetensors','loras.2.strength':.25,'loras.2.enabled':True,
            'loras.1.lora_name':'disabled.safetensors','loras.1.strength':1,'loras.1.enabled':False,
            'loras.0.lora_name':'zero.safetensors','loras.0.strength':0,'loras.0.enabled':True})
        p['sampler']['inputs']['model']=['stack',0]
        self.assertEqual(info(p)['loras'],'<lora:repeat:0.25> <lora:repeat:-0.001>')
        self.assertEqual([r['row'] for r in report(p).fields['lora_entries'].value],[2,10])
        p['stack']['inputs']['loras.2.enabled']=['unknown',0]
        self.assertNotIn('0.25',info(p).get('loras',''))

    def test_clip_only_and_different_strength_not_misrepresented(self):
        p=base();p['lora']=n('LoraLoader',model=['model',0],clip=['clip',0],lora_name='both.safetensors',strength_model=.5,strength_clip=.7)
        p['sampler']['inputs']['model']=['lora',0];p['positive']['inputs']['clip']=['lora',1]
        self.assertNotIn('loras',info(p));self.assertEqual(len(report(p).fields['lora_entries'].value),2)
        p['lora']['inputs']['strength_clip']=.5;self.assertEqual(info(p)['loras'],'<lora:both:0.5>')
        p['lora']['inputs']['strength_model']=0;self.assertNotIn('loras',info(p));self.assertEqual(report(p).fields['lora_entries'].value[0]['target'],'clip')


class ProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.environ.get('SAVE_META_PROVIDER_SOURCES')
        if not root:
            raise unittest.SkipTest('Pinned provider sources not supplied (required in release CI)')
        cls.root=Path(root)
        def load(name,path,package=False):
            spec=importlib.util.spec_from_file_location(name,path,submodule_search_locations=[str(path.parent)] if package else None)
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
        cls.bbox=load('verified_bbox_source',cls.root/'bbox/nodes_element_framing.py')
        cls.pose=load('verified_pose_source',cls.root/'pose/nodes_pose_gesture_injector.py')
        cls.h3=load('verified_h3_source',cls.root/'h3/h3_character_sheet/__init__.py',True)
        cls.old=sys.modules.get('nodes');sys.modules['nodes']=types.SimpleNamespace(MAX_RESOLUTION=16384,NODE_CLASS_MAPPINGS={
            **cls.bbox.NODE_CLASS_MAPPINGS,**cls.pose.NODE_CLASS_MAPPINGS,'H3CharacterSheetDesigner':sys.modules['verified_h3_source.node'].H3CharacterSheetDesigner})

    @classmethod
    def tearDownClass(cls):
        if cls.old is None:sys.modules.pop('nodes',None)
        else:sys.modules['nodes']=cls.old

    def graph(self):
        p=base()
        p['canvas']=n('Krea2ElementFramingV1Canvas',width=512,height=768,preset='Custom',
                       layout_data=json.dumps({'boxes':[{'slot':'red','bbox':[0,0,400,600],'z':1}]}),camera_data='{}',grid_mode='Thirds')
        p['pro']=n('Krea2ElementFramingV1Prompt',scene='scene',background='background',
                   **{s+'_prompt':'main subject' if s=='red' else '' for s in ('red','blue','yellow','green','magenta')})
        p['export']=n('Krea2ElementJSONExportV1',framing_data=['canvas',0],prompt_ui_data=['pro',0],bbox_mode='normalized_1000',skip_empty=True,auto_position_hint=True)
        p['fx']=n('Krea2BBOXPromptEffect',prompt_in=['export',0],enable_effect=True,category='Photo',preset='35mm Film',mode='Preset',custom_preset='',style_boost='Photo')
        p['positive']['inputs']['text']=['fx',0]
        return p

    def expected(self,p):
        m=self.bbox
        c=m.Krea2ElementFramingV1Canvas().execute(**p['canvas']['inputs'])[0]
        pro=m.Krea2ElementFramingV1Prompt().execute(**p['pro']['inputs'])[0]
        export=m.Krea2ElementJSONExportV1().execute(**{**p['export']['inputs'],'framing_data':c,'prompt_ui_data':pro})[0]
        return m.Krea2BBOXPromptEffect().execute(**{**p['fx']['inputs'],'prompt_in':export})

    def test_bbox_complete_chain_matches_actual_compiler(self):
        p=self.graph(); expected=self.expected(p);before=copy.deepcopy(p)
        self.assertEqual(info(p)['prompt'],expected[0]);self.assertEqual(p,before)
        self.assertEqual(json.loads(expected[0].split('\n')[0])['aspect_ratio'],'2:3')

    def test_effect_slots_modes_off_empty_and_unknown(self):
        for mode,preset,enabled in [('Preset','35mm Film',True),('Custom','Realistic Photo',True),('Custom Preset','Custom Preset',True),('Preset','35mm Film',False)]:
            for slot in (0,1):
                with self.subTest(mode=mode,enabled=enabled,slot=slot):
                    p=self.graph();p['fx']['inputs'].update(mode=mode,preset=preset,enable_effect=enabled,custom_preset='chosen effect')
                    p['positive']['inputs']['text']=['fx',slot];self.assertEqual(info(p)['prompt'],self.expected(p)[slot])
        p['fx']['inputs'].update(mode='Preset',preset='nonexistent',enable_effect=True)
        self.assertNotIn('prompt',info(p))

    def test_fragment_does_not_need_upstream_and_export_slot_types(self):
        p=self.graph();p['fx']['inputs']['prompt_in']=['unknown',0];p['positive']['inputs']['text']=['fx',1]
        self.assertIn('prompt',info(p));p['positive']['inputs']['text']=['fx',0];self.assertNotIn('prompt',info(p))
        p['positive']['inputs']['text']=['export',1];self.assertNotIn('prompt',info(p))
        r=Resolver(Graph(prompt=p),VerifiedProviders());self.assertEqual(r.resolve(['export',1],'scalar').value,512)
        self.assertEqual(r.resolve(['export',2],'scalar').value,768)

    def test_bbox_modes_empty_quoted_clipping_and_overrides(self):
        for mode in ['pixel','normalized_1000']:
            for skip in (True,False):
                p=self.graph();p['canvas']['inputs'].update(width_override=1024)
                p['canvas']['inputs']['layout_data']=json.dumps({'boxes':[{'slot':'blue','bbox':[-1,-2,1300,800],'z':2}, {'slot':'red','bbox':[0,0,50,50],'z':1}]})
                p['pro']['inputs']['red_prompt']='"HELLO"';p['export']['inputs'].update(bbox_mode=mode,skip_empty=skip)
                self.assertEqual(info(p)['prompt'],self.expected(p)[0])

    def test_background_custom_and_bundle_precedence(self):
        for mode,custom in [('Preset',''),('Preset','literal custom'),('Custom','')]:
            p=self.graph();p['back']=n('Krea2BackgroundEffect',prompt_in=['export',0],enable_effect=True,category='Background',preset='Classroom',mode=mode,custom_preset=custom,style_boost='Anime')
            p['positive']['inputs']['text']=['back',0]
            r=Resolver(Graph(prompt=p),VerifiedProviders());ex=r.resolve(['export',0],'string').value
            expected=self.bbox.Krea2BackgroundEffect().execute(**{**p['back']['inputs'],'prompt_in':ex})[0]
            self.assertEqual(info(p)['prompt'],expected)

    def test_unknown_mini_box_and_missing_provider_preserve_raw_input(self):
        p=self.graph();p['mini']=n('UnknownMiniBBOX',scene='not the actual compiled text')
        p['positive']['inputs']['text']=['mini',0];self.assertNotIn('prompt',info(p))
        p['sheet']=n('H3CharacterSheetDesigner',state_json='{"not":"final prompt"}')
        p['positive']['inputs']['text']=['sheet',0]
        with patch.dict(sys.modules,{'nodes':types.SimpleNamespace(NODE_CLASS_MAPPINGS={},MAX_RESOLUTION=16384)}):
            self.assertNotIn('prompt',info(p));self.assertEqual(info(p)['seed'],42)

    def test_bbox_background_then_effect_chain(self):
        p=self.graph();p['back']=n('Krea2BackgroundEffect',prompt_in=['export',0],enable_effect=True,category='Background',preset='Classroom',mode='Preset',custom_preset='')
        p['fx']['inputs']['prompt_in']=['back',0]
        r=Resolver(Graph(prompt=p),VerifiedProviders());ex=r.resolve(['export',0],'string').value
        b=self.bbox.Krea2BackgroundEffect().execute(**{**p['back']['inputs'],'prompt_in':ex})[0]
        e=self.bbox.Krea2BBOXPromptEffect().execute(**{**p['fx']['inputs'],'prompt_in':b})[0]
        self.assertEqual(info(p)['prompt'],e)

    def test_source_mismatch_does_not_execute_provider(self):
        p=self.graph();v=VerifiedProviders();v.manifest['nodes']['Krea2BBOXPromptEffect']['files']['nodes_element_framing.py']='0'*64
        r=Resolver(Graph(prompt=p),v).report('save');self.assertNotIn('prompt',to_legacy_info(r));self.assertEqual(to_legacy_info(r)['seed'],42)
        with patch.dict(sys.modules,{'nodes':types.SimpleNamespace(NODE_CLASS_MAPPINGS={},MAX_RESOLUTION=16384)}):
            self.assertNotIn('prompt',info(p))

    def test_h3_designer_matches_pure_compile_all_outputs(self):
        module=sys.modules['verified_h3_source.node'];state=module.DEFAULT_STATE_JSON
        p=base('SamplerCustomAdvanced');p['sheet']=n('H3CharacterSheetDesigner',state_json=state)
        p['positive']=n('MiniMaxH3ReferenceToVideo',prompt=['sheet',0],clip=['clip',0],width=['sheet',1],height=['sheet',2],length=5)
        expected=module.compile_state(state,max_resolution=16384)
        r=Resolver(Graph(prompt=p),VerifiedProviders())
        self.assertEqual(to_legacy_info(r.report('save'))['prompt'],expected['prompt'])
        self.assertEqual(r.resolve(['sheet',1],'scalar').value,expected['width'])
        self.assertEqual(r.resolve(['sheet',2],'scalar').value,expected['height'])
        p['sheet']['inputs']['state_json']='{broken';self.assertNotIn('prompt',info(p));self.assertIn('seed',info(p))

    def test_pose_output_uses_modified_slot(self):
        import inspect
        p=self.graph();cls=self.pose.Krea2BBOXPoseGestureInjectorV1
        args={}
        for k,v in inspect.signature(cls.execute).parameters.items():
            if k=='self':continue
            if v.default is not inspect.Parameter.empty:args[k]=v.default
        schema=cls.INPUT_TYPES()
        for section in ('required','optional'):
            for k,v in schema.get(section,{}).items():
                if k in args:continue
                args[k]=v[1].get('default') if len(v)>1 and isinstance(v[1],dict) and 'default' in v[1] else v[0][0] if isinstance(v[0],list) else ''
        args['prompt_ui_data']=['pro',0]
        args['base_pose_preset']=next(x for x in self.pose.BASE_POSE_PRESET_OPTIONS if x!=self.pose.NONE_OPTION)
        args['enable_pose_preset']=True
        args['target_slot']='RED slot'
        p['pose']=n('Krea2BBOXPoseGestureInjectorV1',**args)
        r=Resolver(Graph(prompt=p),VerifiedProviders());pro=r.resolve(['pro',0],'prompt_data').value
        expected=cls().execute(**{**args,'prompt_ui_data':pro})
        expected=expected['result'] if isinstance(expected,dict) else expected
        self.assertEqual(r.resolve(['pose',0],'prompt_data').value,expected[0])
        self.assertEqual(r.resolve(['pose',1],'string').value,expected[1])
        self.assertEqual(r.resolve(['pose',2],'string').value,expected[2])
        self.assertTrue(expected[1]);self.assertNotEqual(expected[0],pro)
        p['export']['inputs']['prompt_ui_data']=['pose',0]
        canvas=self.bbox.Krea2ElementFramingV1Canvas().execute(**p['canvas']['inputs'])[0]
        export=self.bbox.Krea2ElementJSONExportV1().execute(**{**p['export']['inputs'],'framing_data':canvas,'prompt_ui_data':expected[0]})[0]
        final=self.bbox.Krea2BBOXPromptEffect().execute(**{**p['fx']['inputs'],'prompt_in':export})[0]
        self.assertEqual(info(p)['prompt'],final)


if __name__=='__main__':
    unittest.main()
