import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.pc_brain.build_windows_bundle import build


class WindowsBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.model=root/'field.onnx';self.model.write_bytes(b'fake-private-weights')
        self.apk=root/'hud.apk';self.apk.write_bytes(b'fake-private-apk')
        self.out=root/'delivery.zip'
        self.model_hash=hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.apk_hash=hashlib.sha256(self.apk.read_bytes()).hexdigest()

    def test_wrong_model_or_apk_never_produces_a_delivery(self):
        for model_hash,apk_hash in [('0'*64,self.apk_hash),(self.model_hash,'0'*64)]:
            with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):
                build(self.model,self.apk,self.out,model_hash,apk_hash,'test')
            self.assertFalse(self.out.exists())

    def test_complete_bundle_has_correct_artifacts_and_no_training_assets(self):
        first=build(self.model,self.apk,self.out,self.model_hash,self.apk_hash,'test')
        second=build(self.model,self.apk,self.out,self.model_hash,self.apk_hash,'test')
        self.assertEqual(first['sha256'],second['sha256'])
        with zipfile.ZipFile(self.out) as z:
            manifest=json.loads(z.read('bundle_manifest.json'))
            self.assertEqual(z.read('models/field640.onnx'),self.model.read_bytes())
            self.assertEqual(z.read('efootphy-pc-hud-v0.2.1.apk'),self.apk.read_bytes())
            for name,digest in manifest['files_sha256'].items():
                self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest)
            self.assertIn('01_CAI_DAT_PC.bat',z.namelist())
            self.assertIn('02_CHAY_TRO_LY.bat',z.namelist())
            for name in z.namelist():
                self.assertFalse(name.startswith(('datasets/','legacy/','.venv/')))
                self.assertNotIn('test_',name)
            script=z.read('02_CHAY_TRO_LY.bat')
            self.assertIn(b'EFOOTPHY_EXECUTION_PROVIDER=cpu',script)
            self.assertIn(b'EFOOTPHY_SHADOW_EVIDENCE=0',script)
            self.assertNotIn(b'\n',script.replace(b'\r\n',b''))


if __name__=='__main__':unittest.main()
