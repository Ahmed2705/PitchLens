import unittest
from unittest.mock import patch
import numpy as np
from pitchlens.assisted_team import AssistedTeamClassifier, AssistedConfig
from pitchlens.tracking import Track, FrameTracks


def encoder(crops):
    return np.array([[1.,0.] if x[0,0,0] > 200 else [0.,1.] for x in crops], np.float32)

class AssistedTests(unittest.TestCase):
    def setUp(self):
        self.references = {1:[np.full((40,20,3),255,np.uint8)],2:[np.zeros((40,20,3),np.uint8)]}
        self.classifier = AssistedTeamClassifier(encoder,self.references,AssistedConfig(sample_interval=1,min_observations=2))
        self.frame = np.full((100,100,3),255,np.uint8)
        self.player = Track((10,10,30,70),0,1,.9,True)
        self.keeper = Track((10,10,30,70),1,1,.9,True)

    def test_history_and_cut(self):
        def frame(i,seg=0,cut=False):return FrameTracks(i,seg,cut,(self.player,self.keeper),())
        self.assertEqual(self.classifier.update(self.frame,frame(0)),[None,None])
        self.assertEqual(self.classifier.update(self.frame,frame(1)),[1,None])
        self.assertEqual(self.classifier.update(self.frame,frame(2,1,True)),[None,None])
        self.classifier.reset()
        self.assertEqual(self.classifier.history,{})

    def test_ambiguity_rejected(self):
        self.classifier.prototypes = np.array([[1.,0.],[.9999,.01]],np.float32)
        self.classifier.update(self.frame,FrameTracks(0,0,False,(self.player,),()))
        self.assertEqual(self.classifier.update(self.frame,FrameTracks(1,0,False,(self.player,),())),[None])

    def test_conflicting_observations_are_unknown(self):
        self.classifier.update(self.frame,FrameTracks(0,0,False,(self.player,),()))
        dark=np.zeros_like(self.frame)
        self.assertEqual(self.classifier.update(dark,FrameTracks(1,0,False,(self.player,),())),[None])

    def test_opposite_observations_do_not_crash(self):
        def opposite(crops):
            return np.array([[1.,0.] if x[0,0,0] else [-1.,0.] for x in crops])
        classifier=AssistedTeamClassifier(opposite,self.references,AssistedConfig(sample_interval=1,smoothing=.5))
        self.assertEqual(classifier.update(self.frame,FrameTracks(0,0,False,(self.player,),())),[None])
        self.assertEqual(classifier.update(np.zeros_like(self.frame),FrameTracks(1,0,False,(self.player,),())),[None])

    def test_infeasible_reference_margin(self):
        a=np.array([1.,0.]);b=np.array([np.cos(.025),np.sin(.025)])
        with self.assertRaises(ValueError):
            AssistedTeamClassifier(lambda xs: np.array(xs), {1:[a],2:[b]})

    def test_missing_model_weights_rejected(self):
        from pitchlens.assisted_team import SiglipEncoder
        try:
            import transformers
        except ImportError:
            self.skipTest('optional assisted dependencies are not installed')
        with patch.object(transformers.AutoImageProcessor,'from_pretrained'), \
             patch.object(transformers.SiglipVisionModel,'from_pretrained',return_value=(object(),{'missing_keys':['weight'],'mismatched_keys':[]})):
            with self.assertRaises(ValueError): SiglipEncoder('incomplete-snapshot')

    def test_bad_references(self):
        with self.assertRaises(ValueError): AssistedTeamClassifier(encoder,{1:self.references[1]})
        with self.assertRaises(ValueError): AssistedTeamClassifier(encoder,{1:self.references[1],2:self.references[1]})
        with self.assertRaises(ValueError): AssistedConfig(min_margin=float('nan'))
        with self.assertRaises(ValueError): AssistedConfig(sample_interval=11,max_age=10)

if __name__ == '__main__': unittest.main()
