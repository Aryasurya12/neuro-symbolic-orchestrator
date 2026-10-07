import unittest
from .checker import check,validate_dataset
from .adapters import snapshot_catalog


class CheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=snapshot_catalog()
    def setUp(self):
        self.truth=dict(expected_status='feasible',archetype='vm',rationale='test fixture',constraints=dict(providers=['AWS'],min_vcpus=2,min_ram_gb=4,budget_max_usd=100))
        self.out=dict(allocated_resources={'vms':[dict(provider='AWS',instance_type='t3.medium',count=1)]},claimed_cost_usd=30.37,self_reported_status='feasible')
    def test_valid(self):
        self.assertTrue(check(self.out,self.truth,self.catalog)['passed'])
    def test_claim_cannot_hide_budget_overflow(self):
        self.truth['constraints']['budget_max_usd']=25
        self.out['claimed_cost_usd']=20
        grade=check(self.out,self.truth,self.catalog)
        self.assertFalse(grade['passed']); self.assertIn('budget',grade['reason'])
    def test_fractional_or_nan_or_negative_counts(self):
        for count in [0,-1,0.5,float('nan'),True]:
            self.out['allocated_resources']['vms'][0]['count']=count
            self.assertFalse(check(self.out,self.truth,self.catalog)['passed'])
    def test_provider_identity_checked(self):
        self.out['allocated_resources']['vms'][0]['provider']='GCP'
        self.assertFalse(check(self.out,self.truth,self.catalog)['passed'])
    def test_non_catalog_and_missing_cost(self):
        self.out['allocated_resources']['vms'][0]['instance_type']='made-up'
        self.assertFalse(check(self.out,self.truth,self.catalog)['passed'])
        self.out['allocated_resources']['vms'][0]['instance_type']='t3.medium'
        self.out['claimed_cost_usd']=None
        self.assertFalse(check(self.out,self.truth,self.catalog)['passed'])
    def test_missing_fields_dont_equal_feasible(self):
        t=dict(expected_status='clarification',rationale='missing budget')
        self.assertFalse(check(self.out,t,self.catalog)['passed'])
        self.assertTrue(check(dict(allocated_resources={},claimed_cost_usd=None,self_reported_status='clarification'),t,self.catalog)['passed'])
    def test_pending_prose_is_not_a_failure(self):
        self.assertIsNone(check(None,self.truth,self.catalog)['passed'])
    def test_disaster_recovery_distinct_providers(self):
        t=dict(expected_status='feasible',archetype='dr',constraints=dict(providers=['AWS','Azure'],cross_provider=True,max_latency_ms=100,min_sla_pct=99.99,budget_max_usd=500))
        o=dict(allocated_resources={'regions':['us-east-1','us-west-2']},claimed_cost_usd=266.25,self_reported_status='feasible')
        self.assertIn('cross_provider',check(o,t,self.catalog)['reason'])
        o['allocated_resources']['regions']=['us-east-1','eastus'];o['claimed_cost_usd']=248
        self.assertTrue(check(o,t,self.catalog)['passed'])
    def test_scaling_cpu_independent_of_solver_claim(self):
        t=dict(expected_status='feasible',archetype='scaling',constraints=dict(min_bandwidth_mbps=100,max_bandwidth_mbps=1000,max_replicas=16,max_cpu_pct=70,budget_max_usd=500))
        o=dict(allocated_resources=dict(bandwidth_mbps=100,replicas=1),claimed_cost_usd=53,self_reported_status='feasible')
        self.assertIn('cpu',check(o,t,self.catalog)['reason'])
        o['allocated_resources']['replicas']=2;o['claimed_cost_usd']=98
        self.assertTrue(check(o,t,self.catalog)['passed'])
    def test_unsupported_constraint_rejected_at_freeze(self):
        self.truth['constraints']['gpu_count']=2
        d=dict(ground_truth_author='test',approved_by_user=True,queries=[dict(query_id='dev',query_category='formal_english',raw_query_text='fixture',trials=[1],truth=self.truth)])
        with self.assertRaises(ValueError):validate_dataset(d)


if __name__=='__main__':unittest.main()
