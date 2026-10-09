import sys
sys.path.insert(0, ".")
import re
from src.semantic.normalizer import OutputNormalizer

text = """
Aapki requirements ke mutabiq (AWS aur GCP multi-cloud DR, 99.99% uptime, <50ms latency, budget $600):

Recommended Architecture:
- Primary Region: AWS us-east-1 (N. Virginia)
- Secondary Region: GCP us-central1 (Iowa)
- Cross-Region Latency: ~38ms (Inter-cloud direct fiber route)
- Redundancy Model: Active-Passive Warm Standby

Estimated Monthly Cost Breakdown:
- AWS Base Compute + Storage: $135.00/month
- GCP DR Standby Instance: $88.00/month
- Data Replication Bandwidth: $20.00/month
- Total Estimated Cost: $243.00/month
"""

res = OutputNormalizer.normalize_mode1_prose(text, problem_type="Z3_Graph_Disaster_Recovery")
print("Status:", res[0])
print("Extracted cost:", res[1])
print("Decision:", res[2])
print("Errors:", res[3])
print("Evidence:", [e.to_dict() for e in res[4]])
