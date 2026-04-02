# Google Cloud / Verrtex AI 相关文档

Google Cloud 提供了丰富的官方示例代码和文档，详细介绍了如何在 Vertex AI 上进行分布式训练。这些示例都是为了帮助用户更好地理解和实践，而不是模型生成的通用代码。

以下是一些关键的官方示例和文档资源，您可以参考：

## 1. Vertex AI 官方示例 GitHub 仓库:
GoogleCloudPlatform/vertex-ai-samples 是一个包含大量 Vertex AI 示例的官方 GitHub 仓库，其中包含了许多关于自定义训练、容器化和分布式训练的 Notebook。

+ 多节点分布式数据并行训练 (PyTorch + Custom Container) 示例：这个 Notebook 详细展示了如何使用自定义容器在 Vertex AI 上进行 PyTorch 的多节点分布式数据并行 (DDP) 训练。
  - GitHub 链接： vertex-ai-samples/notebooks/official/training/multi_node_ddp_gloo_vertex_training_with_custom_container.ipynb 
  - 特点： 提供了完整的 Dockerfile、训练脚本以及使用 Vertex AI SDK for Python 提交作业的代码。
+ Vertex AI 分布式训练入门 (TensorFlow/PyTorch)：这个 Notebook 提供了 Vertex AI 分布式训练的通用入门示例。
  - GitHub 链接： vertex-ai-samples/notebooks/official/training/get_started_with_vertex_distributed_training.ipynb 
+ 使用 Reduction Server 优化分布式训练 (PyTorch/TensorFlow)：Reduction Server 是 Vertex AI 优化大规模分布式训练性能的工具。此示例展示了如何将其与 PyTorch 或 TensorFlow 结合使用。
  - GitHub 链接： vertex-ai-samples/notebooks/community/reduction_server/distributed-training-reduction-server.ipynb 

## 2. Google Cloud 官方文档:

+ Vertex AI 训练文档 - 分布式训练：这是 Vertex AI 官方文档中关于分布式训练的概览页面，解释了分布式训练的架构、代码要求以及如何配置。
  - 文档链接： [Vertex AI 官方文档 - 分布式训练](https://docs.cloud.google.com/vertex-ai/docs/training/distributed-training?_gl=1*1th1dj8*_ga*ODA4MzYyOTg2LjE3NzQzMDk3OTU.*_ga_WH2QY8WWF5*czE3NzUwNTE5NDEkbzI3JGcxJHQxNzc1MDUxOTg4JGoxMyRsMCRoMA..) 
  - 特点： 详细说明了 CLUSTER_SPEC 和 TF_CONFIG 等环境变量在分布式训练中的作用，以及不同的工作器池角色（Primary、Worker、Parameter Server、Evaluator）。
+ 创建自定义容器镜像进行训练：如果您需要为分布式训练构建自定义 Docker 镜像，这个文档提供了详细的指南和 Dockerfile 示例，包括 GPU 支持的配置。
  - 文档链接： [创建自定义容器镜像进行训练](https://cloud.google.com/vertex-ai/docs/training/create-custom-container?hl=en_US&_gl=1*6cqm5r*_ga*ODA4MzYyOTg2LjE3NzQzMDk3OTU.*_ga_WH2QY8WWF5*czE3NzUwNTE5NDEkbzI3JGcxJHQxNzc1MDUyMjU3JGo1OCRsMCRoMA..) 
  - 特点： 包含针对 GPU 训练的 Dockerfile 调整，例如使用 nvidia/cuda 作为基础镜像。
+ 配置 Vertex AI Serverless 训练的计算资源：这个文档详细介绍了如何在 Vertex AI Serverless 训练中配置机器类型、GPU 类型和数量等计算资源。
  - 文档链接： [配置 Vertex AI Serverless 训练的计算资源](https://cloud.google.com/vertex-ai/docs/training/configure-compute?hl=en_US&_gl=1*6cqm5r*_ga*ODA4MzYyOTg2LjE3NzQzMDk3OTU.*_ga_WH2QY8WWF5*czE3NzUwNTE5NDEkbzI3JGcxJHQxNzc1MDUyMjU3JGo1OCRsMCRoMA..) 
  - 特点： 包含使用 gcloud CLI 和 Python SDK 配置加速器的示例。
+ Vertex AI 最佳实践 - 准备训练代码：此文档提供了编写训练代码的最佳实践，以便更好地在 Vertex AI 上运行，尤其是在使用自定义容器时。
  - 文档链接： [Vertex AI - 准备训练代码](https://cloud.google.com/vertex-ai/docs/training/code-requirements?hl=en_US&_gl=1*5a7tfk*_ga*ODA4MzYyOTg2LjE3NzQzMDk3OTU.*_ga_WH2QY8WWF5*czE3NzUwNTE5NDEkbzI3JGcxJHQxNzc1MDUyMjU3JGo1OCRsMCRoMA..)
 
## 3. Google Codelabs:

+ Vertex AI: Multi-Worker Training and Transfer Learning with TensorFlow: 这是一个 Codelab，通过实践指导您如何使用 Vertex AI 运行多工作器（分布式）训练作业。
  - Codelabs 链接： [Vertex AI: Multi-Worker Training and Transfer Learning with TensorFlow](https://www.google.com/url?q=https%3A%2F%2Fcodelabs.developers.google.com%2Fvertex-ai-multi-worker-tensorflow)
 
## 4. 如何使用这些资源：

1, 从 GitHub 示例开始： 建议您首先查看 vertex-ai-samples 仓库中的相关 Notebook。它们通常是端到端的示例，包含了代码、说明和部署步骤。您可以直接在 Colab 或 Vertex AI Workbench 中打开并运行这些 Notebook。
2, 参考官方文档： 在理解示例代码的基础上，查阅官方文档可以更深入地了解每个配置选项的含义和背后的原理。
3, 实践 Codelabs： Codelabs 提供了一种交互式、分步指导的学习体验，非常适合从头开始构建和部署分布式训练作业。
这些官方资源将为您在 Vertex AI 上部署分布式训练作业提供坚实的基础和实践指导。

## 5. 其它资源

+ https://docs.cloud.google.com/blog/topics/developers-practitioners/streamline-your-ml-training-workflow-vertex-ai 
+ https://docs.cloud.google.com/blog/topics/developers-practitioners/cloud-ai-developer-community 
+ https://stackoverflow.com/questions/46615264/does-google-cloud-ml-only-support-distributed-tensorflow-for-multiple-gpu-traini 
