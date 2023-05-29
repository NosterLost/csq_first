import lightgbm as lgb
import pandas as pd
import numpy as np

label = pd.read_csv('test_label.csv', index_col=0)

# 特征最后列数
feature_num = 190

# 用训练好的模型预测未标签的数据
def lgb_train_2():
    # 导入模型
    gbm = lgb.Booster(model_file='trained_lgb_model.txt')

    # 导入待测数据
    df_test = pd.read_csv('unlabeled_test.csv')
    test_data = df_test.iloc[:, 2:feature_num].values
    test_target = df_test['family_no'].values

    y_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
    print(len(y_pred))
    print(y_pred)
    return y_pred


# 二分类训练结果导入文件test2.csv
def lgb_train2_label(y_pred):
    group = pd.DataFrame({'family_no': y_pred})
    group.reset_index(inplace=True)
    group.to_csv('test.csv', index=False)

    y_pred2 = np.where(y_pred > 0.5, 1, 0)
    group = pd.DataFrame({'family_no': y_pred2})
    # 与fqdn_no列合并
    data = pd.read_csv('unlabeled_test.csv', index_col=0)
    domain = data['fqdn_no']
    regroup = [domain, group]
    regroup = pd.concat(regroup, axis=1)
    # 删去0, 得到二分类结果
    regroup = regroup[regroup['family_no'] != 0]
    print('lgb_2 :', len(regroup))
    regroup.reset_index(inplace=True)
    regroup = regroup.drop(['index'], axis=1)
    regroup.to_csv('test_2_lgb.csv')


# 用model_lgbn多分类训练
def lgb_train_n():
    gbm = lgb.Booster(model_file='model_lgbn.txt')

    # 导入待测数据
    df_test = pd.read_csv('test_2_lgb.csv', index_col=0)
    df_all_feature = pd.read_csv('feature_all.csv', index_col=0)

    data = pd.merge(df_test, df_all_feature, on='fqdn_no', how='left')
    data.reset_index(inplace=True)
    data = data.drop(['family_no', 'index'], axis=1)
    data.to_csv('test_undivided_lgb.csv')

    # 训练
    test_data = pd.read_csv('test_undivided_lgb.csv')
    test_data = test_data.iloc[:, 2:feature_num].values

    y_prob = gbm.predict(test_data, num_iteration=gbm.best_iteration)
    y_pred = [list(x).index(max(x)) for x in y_prob]
    return y_pred


# 恶意域名多分类训练结果输出
def lgb_train_n_label(y_pred):

    group = pd.DataFrame({'family_no': y_pred})
    group.reset_index(inplace=True)
    data = pd.read_csv('test_2_lgb.csv', index_col=0)
    # 合并
    data = data.drop(['family_no'], axis=1)
    regroup = [data, group]
    regroup = pd.concat(regroup, axis=1)
    regroup.reset_index(inplace=True)
    regroup = regroup.drop(['index', 'level_0'], axis=1)
    regroup.to_csv('test_divided_lgb.csv')


    regroup = pd.read_csv('test_divided_lgb.csv', index_col=0)
    regroup = pd.concat([regroup, label])
    regroup.sort_values(['fqdn_no'], inplace=True)
    regroup.to_csv('test_divided_lgb.csv', index=False)
    print(regroup['family_no'].value_counts())

if __name__ == "__main__":
    # 黑白二分类
    y_pred_2 = lgb_train_2()
    lgb_train2_label(y_pred_2)

    # 恶意域名家族多分类
    y_pred_n = lgb_train_n()
    lgb_train_n_label(y_pred_n)
