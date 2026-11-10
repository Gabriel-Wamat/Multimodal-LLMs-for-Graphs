import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.models.control.hardware import validate_gpu


class HardwareTests(unittest.TestCase):
    def test_rejects_mig_allocation_before_loading_weights(self):
        cuda = SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,
                               get_device_properties=lambda _:SimpleNamespace(total_memory=10*1024**3))
        with patch.dict(sys.modules, {'torch':SimpleNamespace(cuda=cuda)}):
            with self.assertRaisesRegex(RuntimeError,'at least 20'):
                validate_gpu('llava',{'inference_min_vram_gib':{'llava':20}})

    def test_training_uses_its_own_requirement(self):
        cuda = SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:True,
                               get_device_properties=lambda _:SimpleNamespace(total_memory=24*1024**3))
        config={'inference_min_vram_gib':{'llava':20},'training_min_vram_gib':{'llava':48}}
        with patch.dict(sys.modules, {'torch':SimpleNamespace(cuda=cuda)}):
            validate_gpu('llava',config)
            with self.assertRaisesRegex(RuntimeError,'at least 48'):
                validate_gpu('llava',config,training=True)

    def test_no_cuda_is_an_explicit_error(self):
        with patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))}):
            with self.assertRaisesRegex(RuntimeError,'CUDA'):
                validate_gpu('qwen',{})
