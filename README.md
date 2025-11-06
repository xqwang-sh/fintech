# 金融科技导论课程讲义

本项目将金融科技导论课程的幻灯片转换为 Quarto 书籍格式，方便在线发布和分享。

## 项目结构

```
handouts/book/
├── _quarto.yml          # Quarto 书籍配置文件
├── index.qmd           # 书籍首页
├── styles.css          # 自定义样式
├── test.qmd            # 测试文件
├── chapters/           # 章节内容
│   ├── w0_syllabus.qmd              # 第0周：课程安排与考核要求
│   ├── w1_python_ml_intro.qmd       # 第1周：Python基础与机器学习入门
│   ├── w2_data_processing.qmd       # 第2周：数据处理与线性回归实践
│   ├── w3_classification_decision_tree.qmd  # 第3周：分类问题、决策树与评估指标
│   ├── w4_ensemble_learning.qmd     # 第4周：集成学习 - 随机森林与 GBDT
│   ├── w5_neural_networks.qmd       # 第5周：神经网络基础 - MLP 入门
│   └── w6_neural_networks_advanced.qmd  # 第6周：神经网络进阶 - 正则化与调优
├── images/             # 图片资源目录
└── references.qmd      # 参考资料
```

## 构建书籍

### 环境要求

1. **安装 Quarto**
   ```bash
   # macOS
   brew install quarto

   # 或从官网下载: https://quarto.org/docs/get-started/
   ```

2. **安装 Python 依赖**（如果要执行代码块）
   ```bash
   pip install jupyter matplotlib seaborn pandas scikit-learn tensorflow lightgbm
   ```

### 构建命令

#### 构建 HTML 版本（推荐）
```bash
quarto render
```

#### 构建 PDF 版本
```bash
quarto render --to pdf
```

#### 构建并预览
```bash
quarto preview
```

### 常见问题

#### 1. Chrome Headless 模式错误
如果遇到 Chrome 相关错误，请修改 `_quarto.yml` 中的执行配置：

```yaml
execute:
  eval: false  # 禁用代码执行
```

#### 2. 代码块不执行
书籍版本默认禁用了代码执行，以避免构建问题。如需执行代码：

```yaml
execute:
  eval: true   # 启用代码执行
```

#### 3. 图片显示问题
确保图片路径正确，建议将图片放在 `images/` 目录下。

## 书籍内容

### 第一部分：课程概述
- **W0**: 课程安排与考核要求
  - 教学目标、课程结构
  - 考核方式、项目要求

### 第二部分：机器学习基础
- **W1**: Python基础与机器学习入门
  - 机器学习定义、监督学习流程
  - 线性回归、训练集/测试集

- **W2**: 数据处理与线性回归实践
  - 数据预处理（缺失值、异常值、归一化）
  - 回归评估指标（MAE、RMSE、R²）
  - 残差分析

- **W3**: 分类问题、决策树与评估指标
  - 分类 vs 回归
  - 决策树原理、混淆矩阵
  - 精确率、召回率、F1、AUC

- **W4**: 集成学习 - 随机森林与 GBDT
  - Bagging vs Boosting
  - 随机森林、特征重要性
  - GBDT、超参数调整

### 第三部分：深度学习与高级主题
- **W5**: 神经网络基础 - MLP 入门
  - 神经元模型、激活函数
  - 多层感知机、Keras 实战

- **W6**: 神经网络进阶 - 正则化与调优
  - 过拟合诊断、正则化技术
  - 学习率调度、模型对比

## 部署到网上

### GitHub Pages
1. 将项目推送到 GitHub
2. 在仓库设置中启用 Pages
3. 选择 `gh-pages` 分支或 `docs/` 文件夹

### Netlify
1. 连接 GitHub 仓库
2. 设置构建命令：`quarto render`
3. 发布目录：`_book`

### Vercel
1. 导入 GitHub 仓库
2. 构建命令：`quarto render`
3. 输出目录：`_book`

## 自定义配置

### 修改书籍信息
编辑 `_quarto.yml` 中的书籍元数据：

```yaml
book:
  title: "你的标题"
  author: "你的名字"
  date: "2025"
```

### 调整样式
修改 `styles.css` 文件来自定义外观。

### 添加新章节
1. 在 `chapters/` 目录下创建新的 `.qmd` 文件
2. 在 `_quarto.yml` 的 `chapters` 列表中添加引用

## 许可证

本书采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 协议。

## 贡献

欢迎提交 Issue 和 Pull Request 来改进本书。

---

**构建成功后，你将得到一个完整的在线书籍，可以通过浏览器访问和分享！** 📚
