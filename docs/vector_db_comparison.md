# Vector DB Comparison: FAISS vs OpenSearch (AWS)

## Overview

For the RAG pipeline, vector databases are needed to store and retrieve embeddings efficiently.  
Here we compare **FAISS** (open-source, local) and **OpenSearch** (cloud-managed).



## Comparison Table

| Feature        | FAISS (Facebook AI Similarity Search)              | OpenSearch (AWS Managed Service)                                   |
| -------------- | -------------------------------------------------- | ------------------------------------------------------------------ |
| **Deployment** | Local / on-premise, runs directly on CPU/GPU       | Fully managed AWS service, integrates with AWS ecosystem  **(includes Serverless Vector DB option)**|
| **Scalability**| Limited by local hardware resources                | Horizontal scaling with elastic cloud infra                        |
| **Performance**| Very fast for single-node search; GPU acceleration | Good for large-scale distributed queries                           |
| **Ecosystem**  | Open-source, widely used in research and prototypes| AWS native (IAM, CloudWatch, Terraform support, VPC integration)   |
| **Ease of Use**| Requires manual setup & maintenance                | Managed service, simple API calls, no infra management             |
| **Cost**       | Free (open-source)                                 | Pay-as-you-go, but credits possible for students                   |
| **Best Use Case** | Local experiments, academic research, prototyping | Production workloads, cloud-native MLOps pipelines                 |


## Project Relevance

- **FAISS** will be used for local prototyping and benchmarking.  
- **OpenSearch** will be adopted in later stages for cloud deployment and full MLOps integration.  

## References

- **FAISS**
  - Official GitHub: [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
  - Documentation: [https://faiss.ai/](https://faiss.ai/)

- **OpenSearch (AWS)**
  - OpenSearch Project Official: [https://opensearch.org/](https://opensearch.org/)
  - AWS OpenSearch Serverless Vector DB: https://aws.amazon.com/opensearch-service/serverless-vector-database/
  
