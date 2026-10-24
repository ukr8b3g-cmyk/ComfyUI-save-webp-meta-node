"""Reported zero-strength, runtime filename and dry-run schema regressions."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from test_metadata_resolver import base, n, info, report
from test_webp_save import webp_save


class LoraBoundaryTests(unittest.TestCase):
    def graph(self, model, clip):
        p = base()
        p['lora'] = n('LoraLoader', model=['model', 0], clip=['clip', 0],
                      lora_name='both.safetensors', strength_model=model, strength_clip=clip)
        p['sampler']['inputs']['model'] = ['lora', 0]
        p['positive']['inputs']['clip'] = ['lora', 1]
        return p

    def test_strength_matrix_and_raw_graph_preserved(self):
        for model, clip, tag in [(.5, 0, None), (0, .5, None), (0, 0, None),
                                 (.5, .5, '<lora:both:0.5>'), (.5, .7, None),
                                 (-.5, -.5, '<lora:both:-0.5>')]:
            with self.subTest(model=model, clip=clip):
                p = self.graph(model, clip); before = copy.deepcopy(p)
                self.assertEqual(info(p).get('loras'), tag)
                self.assertEqual(p, before)
                r = report(p)
                entries = r.fields.get('lora_entries')
                self.assertTrue(all(row.get('strength') != 0 for row in (entries.value if entries else [])))
                self.assertEqual(len(r.fields['lora_targets'].value), 2)

    def test_unconnected_clip_is_distinct_from_zero(self):
        p = self.graph(.5, 0); p['positive']['inputs']['clip'] = ['clip', 0]
        self.assertEqual(info(p)['loras'], '<lora:both:0.5>')

    def test_unresolved_clip_strength_cannot_be_a_single_tag(self):
        p = self.graph(.5, ['unknown', 0])
        self.assertNotIn('loras', info(p))

    def test_saved_metadata_omits_inaccurate_tag_and_keeps_raw_graph(self):
        from test_metadata_save_formats import SaveFormatTests, Tensor
        helper = SaveFormatTests(); helper.setUp()
        p = self.graph(.5, 0); before = copy.deepcopy(p)
        with tempfile.TemporaryDirectory() as directory:
            for fmt in ['png', 'webp']:
                result = helper.run_save(directory, p, [Tensor()], fmt=fmt)
                parameters, prompt, workflow = helper.read_meta(Path(directory)/result['ui']['images'][0]['filename'])
                self.assertNotIn('<lora:both:', parameters)
                self.assertEqual(prompt, before)
                self.assertEqual(p, before)


class FilenameTests(unittest.TestCase):
    def setUp(self):
        self.p = base(); self.p['sampler']['_meta'] = {'title': 'Selected'}
        self.wf = {'nodes': [{'id': 'sampler', 'type': 'KSampler', 'title': 'Selected',
                             'widgets_values': [999, 'fixed', 20, 1, 'euler', 'simple', 1]}]}
        self.saver = webp_save.SaveWebPMeta()

    def name(self, prompt):
        return self.saver._format_filename('%seed%_%Selected.seed%', {'seed':42}, prompt=prompt, extra_pnginfo={'workflow':self.wf})

    def test_runtime_wins_and_graphs_are_unchanged(self):
        before = copy.deepcopy((self.p, self.wf))
        self.assertEqual(self.name(self.p), '42_42')
        self.assertEqual((self.p, self.wf), before)

    def test_known_link_resolves(self):
        self.p['value'] = n('PrimitiveInt', value=123)
        self.p['sampler']['inputs']['seed'] = ['value', 0]
        self.assertEqual(self.name(self.p), '42_123')

    def test_unknown_missing_and_empty_runtime_never_use_stale_ui(self):
        for value in [['unknown', 0], None]:
            self.p['sampler']['inputs']['seed'] = value
            self.assertEqual(self.name(self.p), '42_%Selected.seed%')
        del self.p['sampler']['inputs']['seed']
        self.assertEqual(self.name(self.p), '42_%Selected.seed%')
        self.assertEqual(self.name({}), '42_%Selected.seed%')

    def test_workflow_only_and_sr_name_resolve(self):
        self.assertEqual(self.name(None), '42_999')
        del self.p['sampler']['_meta']
        self.wf['nodes'][0]['title'] = 'Other'
        self.wf['nodes'][0]['properties'] = {'Node name for S&R': 'Selected'}
        self.assertEqual(self.name(self.p), '42_42')

    def test_duplicate_names_are_unresolved_in_both_sources(self):
        self.p['second'] = copy.deepcopy(self.p['sampler'])
        self.assertEqual(self.name(self.p), '42_%Selected.seed%')
        self.wf['nodes'].append(dict(self.wf['nodes'][0], id='second'))
        self.assertEqual(self.name(None), '42_%Selected.seed%')

    def test_zero_and_64_bit_seed_preserved(self):
        for value in [0, 2**64-1]:
            self.p['sampler']['inputs']['seed'] = value
            self.assertEqual(self.name(self.p), '42_'+str(value))


class SchemaTests(unittest.TestCase):
    def test_schema_contract_and_real_reports(self):
        schema = json.loads((Path(__file__).resolve().parents[1]/'docs/METADATA_RESOLUTION.schema.json').read_text())
        self.assertEqual(set(schema['properties']['image']['type']), {'object', 'null'})
        self.assertEqual(set(schema['properties']['image']['required']), {'width','height','channels','batch_index'})
        try:
            from jsonschema import Draft202012Validator
        except ImportError:
            # Keep the existing dependency-free schema contract check in CI;
            # perform full Draft validation when jsonschema is already available.
            return
        validator = Draft202012Validator(schema)
        self.assertFalse(list(validator.iter_errors(report(base()).as_dict())))
        image = {'width':17,'height':13,'channels':3,'batch_index':0}
        self.assertFalse(list(validator.iter_errors(report(base(), image=image).as_dict())))
        for bad in [{}, dict(image, width=0), dict(image, channels=2)]:
            self.assertTrue(list(validator.iter_errors(report(base(), image=image).as_dict() | {'image': bad})))
