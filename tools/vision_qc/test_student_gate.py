import copy
import unittest

from .student_gate import decide,no_regression


def report(pck=.18,median=12.,tail=230.,coverage=.53,weight=0.):
    return {'validation':{'summary':{'visible_ground_truth_points':92,'pck':{'5':pck},'coverage':coverage,
                    'coordinate_error_on_detected_objects':{'count':92,'median_px':median,'p95_px':tail}}},
            'encoder_sha256':'encoder','source_dataset_fingerprint':'data','point_threshold':.5,
            'coordinate_space':'640x640 source pixels','initial_head_sha256':'initial','optimizer_updates':600,
            'batch':8,'gradient_accumulation':1,'training_views':404,'source_train_images':93,'pseudo_train_images':8,
            'views_per_source':4,'seed':20261101,'teacher_proposals_sha256':'teacher','pseudo_policy':'teacher_only',
            'teacher_weight':weight,'validation_used_for_optimization_or_checkpoint_selection':False}


class StudentGateTests(unittest.TestCase):
    def test_pck_gain_does_not_hide_worse_tail(self):
        result=decide(report(),report(),report(.3,8,300,.6,.25))
        self.assertEqual(result['selected_research_checkpoint'],'baseline')
        self.assertFalse(result['distilled_vs_baseline']['checks']['p95'])

    def test_all_metrics_must_hold_against_baseline_and_control(self):
        result=decide(report(),report(.2,11,220,.54),report(.24,10,210,.55,.25))
        self.assertEqual(result['selected_research_checkpoint'],'distilled');self.assertFalse(result['release_eligible'])
        self.assertFalse(result['android_model_changed'])

    def test_better_control_is_not_replaced_by_inferior_distillation(self):
        result=decide(report(),report(.3,8,180,.6),report(.24,10,210,.55,.25))
        self.assertEqual(result['selected_research_checkpoint'],'control')

    def test_different_data_steps_or_augmentation_are_rejected(self):
        base=report();control=report();student=report(weight=.25)
        for key,value in [('source_dataset_fingerprint','different'),('optimizer_updates',601)]:
            changed=copy.deepcopy(student);changed[key]=value
            with self.assertRaises(ValueError):decide(base,control,changed)
        control['augmentation_records']=[{'source':'train.jpg','view':1,'matrix_320':[[1]],'image_sha256':'a'}]
        student['augmentation_records']=[{'source':'train.jpg','view':1,'matrix_320':[[1]],'image_sha256':'b'}]
        with self.assertRaises(ValueError):decide(base,control,student)

    def test_omitted_coordinates_cannot_improve_p95(self):
        changed=report(.3,1,2,.6,.25);changed['validation']['summary']['coordinate_error_on_detected_objects']['count']=60
        with self.assertRaises(ValueError):no_regression(report(),changed)

    def test_validation_driven_checkpoint_is_rejected(self):
        changed=report(weight=.25);changed['validation_used_for_optimization_or_checkpoint_selection']=True
        with self.assertRaises(ValueError):decide(report(),report(),changed)


if __name__=='__main__':unittest.main()
