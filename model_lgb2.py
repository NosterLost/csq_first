import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# 加载训练数据
df = pd.read_csv('labeled_all.csv')
df.dropna(how='any', inplace=True)  # 去掉有空值的行

data = df.iloc[:, 2:190].values  # 2-列为特征
target = df['family_no'].values  # family_no为预测目标

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=2022, test_size=0.9)

# 从已标签的恶意域名中合并到随机取样本
df_labeled = pd.read_csv('labeled_1.csv')
df_labeled.dropna(how='any', inplace=True)

# 有标签的特征和训练目标
data_labeled = df_labeled.iloc[:, 2:190].values
target_labeled = df_labeled['family_no'].values
X_train = np.vstack((X_train, data_labeled))
y_train = np.hstack((y_train, target_labeled))

# print("Train data length:", len(X_train))
# print("Test data length:", len(X_test))

# 转换为Dataset数据格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'num_leaves': 50,  # 结果对最终效果影响较大，越大值越好，太大会出现过拟合
    'min_data_in_leaf': 60,
    'objective': 'binary',  # 定义的目标函数
    'max_depth': 6,
    'learning_rate': 0.05,
    "boosting": "gbdt",
    "feature_fraction": 0.8,  # 提取的特征比率
    "bagging_freq": 5,
    "bagging_fraction": 0.8,
    'lambda_l1': 0.1,  # l1正则
    'lambda_l2': 0.1,     #l2正则
    'metric': {'auc'},  ##评价函数选择
    "verbosity": -1,
    "nthread": 4,  # 线程数量，-1表示全部线程，线程越多，运行的速度越快
    "random_state": 1902
}

# 训练轮数
boost_round = 200
early_stop_rounds = 100

results = {}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=boost_round,
                valid_sets=[lgb_eval, lgb_train],
                valid_names=['validate', 'train'],
                # early_stopping_rounds=early_stop_rounds,
                evals_result=results)

# 模型保存
#gbm.save_model('trained_lgb_model.txt')

# 模型预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)


# 超过0.5判为1
y_pred2 = np.where(y_pred > 0.5, 1, 0)

print(classification_report(y_test, y_pred2))
acu_score = accuracy_score(y_test, y_pred2)
print("accuracy_score:", acu_score)

# 预测实际结果并绘图
lgb.plot_metric(results)
plt.show()

print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
