import tempfile
from pathlib import Path
import unittest
from nanogpt_noise.input_bundle import register_bundle,verify_bundle
from nanogpt_noise.prepare import fill,prepare_bundle
from tests.test_integration import fixture


class Encoder:
    eot_token=9
    def encode_ordinary(self,text): return [int(value) for value in text.split(',')]


class PreparationTests(unittest.TestCase):
    def test_fill_truncates_boundary_and_discards_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            stream=iter([{'text':'1,2'},{'text':'3,4,5,6'}])
            path=Path(directory)/'tokens.bin'
            result=fill(stream,Encoder(),path,5)
            self.assertEqual(result,dict(tokens=5,documents_consumed=2,
                                         boundary_document_truncated=True))
            import numpy as np
            self.assertEqual(np.fromfile(path,dtype='<u2').tolist(),[1,2,9,3,4])

    def test_register_verify_and_refuse_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data,artifacts=fixture(root)
            data.rename(root/'openwebtext_subset')
            artifacts.rename(root/'frozen-artifacts')
            data,artifacts=root/'openwebtext_subset',root/'frozen-artifacts'
            identity=register_bundle(root,data,artifacts)
            self.assertEqual(verify_bundle(root,data,artifacts),identity)
            self.assertEqual(prepare_bundle(root),identity)
            self.assertTrue((root/'registration.json').is_file())
            self.assertEqual(len((root/'SHA256SUMS').read_text().splitlines()),6)
            with open(data/'train.bin','ab') as stream: stream.write(b'changed')
            with self.assertRaises(ValueError): verify_bundle(root,data,artifacts)

    def test_registration_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data,artifacts=fixture(root)
            data.rename(root/'openwebtext_subset')
            artifacts.rename(root/'frozen-artifacts')
            data,artifacts=root/'openwebtext_subset',root/'frozen-artifacts'
            register_bundle(root,data,artifacts)
            registration=root/'registration.json'
            text=registration.read_text(encoding='utf8').replace('"schema": 1','"schema": 2')
            registration.write_text(text,encoding='utf8')
            with self.assertRaises(ValueError): register_bundle(root,data,artifacts)


if __name__=='__main__': unittest.main()
