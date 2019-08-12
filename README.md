# 概述 
#### X-DeepLearning(简称XDL)是面向高维稀疏数据场景（如广告/推荐/搜索等）深度优化的一整套解决方案。XDL1.2版本已于近期发布，主要特性包括：
* 针对大batch/低并发场景的性能优化：在此类场景下性能提升50-100%
* 存储及通信优化：参数无需人工干预自动全局分配，请求合并，彻底消除ps的计算/存储/通信热点
* 完整的流式训练特性：包括特征准入，特征淘汰，模型增量导出，特征counting统计等  
* Fix了若干1.0中的小bugs  


完整介绍请参考[XDL1.2 release note](https://github.com/alibaba/x-deeplearning/releases/tag/v1.2)

### 1. XDL训练引擎

* [编译安装](https://github.com/alibaba/x-deeplearning/wiki/%E7%BC%96%E8%AF%91%E5%AE%89%E8%A3%85)
* [快速开始](https://github.com/alibaba/x-deeplearning/wiki/%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)
* [使用指南](https://github.com/alibaba/x-deeplearning/wiki/%E7%94%A8%E6%88%B7%E6%96%87%E6%A1%A3)

### 2. XDL算法解决方案
* [快速开始](https://github.com/alibaba/x-deeplearning/wiki/XDL%E7%AE%97%E6%B3%95%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88)

### 3. Blaze预估引擎
* [快速开始](https://github.com/alibaba/x-deeplearning/blob/master/blaze/README.md)

### 4. 深度树匹配模型 TDM 匹配召回引擎 
* [快速开始](https://github.com/alibaba/x-deeplearning/wiki/TDMServing)

# 联系我们
* 欢迎通过[issue](https://github.com/alibaba/x-deeplearning/issues)和邮件组(xdl-opensource@list.alibaba-inc.com
)联系我们
* 我们正在寻求合作伙伴，有志于获得XDL企业级支持计划的公司或团队，可以联系xdl-partner@list.alibaba-inc.com，与我们进一步商谈。

# FAQ
[常见问题](https://github.com/alibaba/x-deeplearning/wiki/FAQ)

# License
XDL使用[Apache-2.0](https://github.com/alibaba/x-deeplearning/blob/master/xdl/LICENSE)许可

# 致谢
XDL项目由阿里妈妈事业部荣誉出品，核心贡献团队包括阿里妈妈工程平台、算法平台、定向广告技术团队、搜索广告技术团队等，同时XDL项目也得到了阿里巴巴计算平台事业部（特别是PAI团队）的帮助。

