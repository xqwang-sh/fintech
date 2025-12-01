# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载 Boston Housing 数据集
df = pd.read_csv('boston_housing.csv')

# 显示数据集的前五行
print(df.head())
# 显示数据集的基本信息
print(df.info())

# 显示描述性统计量
print(df.describe())

# 检查缺失值
print(df.isnull().sum())

# 分离特征和目标变量
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 切分训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对训练集进行标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 可视化特征与目标变量的关系
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X['RM'], y=y)
plt.title('房间数 (RM) 与 房价 (MEDV) 的关系')
plt.xlabel('房间数 (RM)')
plt.ylabel('房价 (MEDV)')
plt.show()

# 线性回归模型训练示例
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 导入评估函数

# 评估模型
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"训练 R^2: {train_r2:.4f}, 测试 R^2: {test_r2:.4f}")
print(f"训练 RMSE: {train_rmse:.4f}, 测试 RMSE: {test_rmse:.4f}")
print(f"训练 MAE: {train_mae:.4f}, 测试 MAE: {test_mae:.4f}")

# ridge 回归模型训练示例
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

y_train_pred_ridge = ridge.predict(X_train_scaled)
y_test_pred_ridge = ridge.predict(X_test_scaled)

train_mse_ridge = mean_squared_error(y_train, y_train_pred_ridge)
test_mse_ridge = mean_squared_error(y_test, y_test_pred_ridge)
train_rmse_ridge = np.sqrt(train_mse_ridge)
test_rmse_ridge = np.sqrt(test_mse_ridge)
train_mae_ridge = mean_absolute_error(y_train, y_train_pred_ridge)
test_mae_ridge = mean_absolute_error(y_test, y_test_pred_ridge)
train_r2_ridge = r2_score(y_train, y_train_pred_ridge)
test_r2_ridge = r2_score(y_test, y_test_pred_ridge)

print(f"Ridge (alpha=1) 训练 R^2: {train_r2_ridge:.4f}, 测试 R^2: {test_r2_ridge:.4f}")
print(f"Ridge (alpha=1) 训练 RMSE: {train_rmse_ridge:.4f}, 测试 RMSE: {test_rmse_ridge:.4f}")
print(f"Ridge (alpha=1) 训练 MAE: {train_mae_ridge:.4f}, 测试 MAE: {test_mae_ridge:.4f}")

# 用CV选择最佳alpha
from sklearn.linear_model import RidgeCV
alphas = np.linspace(0.01, 100, 50)
ridge_cv = RidgeCV(alphas=alphas, store_cv_values=True)
ridge_cv.fit(X_train_scaled, y_train)

best_alpha = ridge_cv.alpha_
print(f"最佳 alpha: {best_alpha}")
y_train_pred_ridge_cv = ridge_cv.predict(X_train_scaled)
y_test_pred_ridge_cv = ridge_cv.predict(X_test_scaled)

train_mse_ridge_cv = mean_squared_error(y_train, y_train_pred_ridge_cv)
test_mse_ridge_cv = mean_squared_error(y_test, y_test_pred_ridge_cv)
train_rmse_ridge_cv = np.sqrt(train_mse_ridge_cv)
test_rmse_ridge_cv = np.sqrt(test_mse_ridge_cv)
train_mae_ridge_cv = mean_absolute_error(y_train, y_train_pred_ridge_cv)
test_mae_ridge_cv = mean_absolute_error(y_test, y_test_pred_ridge_cv)
train_r2_ridge_cv = r2_score(y_train, y_train_pred_ridge_cv)
test_r2_ridge_cv = r2_score(y_test, y_test_pred_ridge_cv)
print(f"Ridge CV (alpha={best_alpha}) 训练 R^2: {train_r2_ridge_cv:.4f}, 测试 R^2: {test_r2_ridge_cv:.4f}")
print(f"Ridge CV (alpha={best_alpha}) 训练 RMSE: {train_rmse_ridge_cv:.4f}, 测试 RMSE: {test_rmse_ridge_cv:.4f}")
print(f"Ridge CV (alpha={best_alpha}) 训练 MAE: {train_mae_ridge_cv:.4f}, 测试 MAE: {test_mae_ridge_cv:.4f}")