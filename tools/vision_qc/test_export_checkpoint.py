import unittest

from .export_checkpoint import exportable_epoch


class ExportMetadataTest(unittest.TestCase):
    def test_saved_epoch_is_converted_from_zero_based_index(self):
        self.assertEqual(exportable_epoch({'epoch':9}),10)

    def test_stripped_epoch_is_unknown_instead_of_fabricated_zero(self):
        self.assertIsNone(exportable_epoch({'epoch':-1}))
        self.assertIsNone(exportable_epoch({}))

    def test_debug_optimizer_steps_cannot_be_mislabeled_as_production_epochs(self):
        with self.assertRaises(ValueError):exportable_epoch({'epoch':-1,'optimization_steps':2096,'debug_overfit_only':True})
