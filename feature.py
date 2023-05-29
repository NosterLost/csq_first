import pandas as pd
import numpy as np
import json
import csv
from math import log
from functools import reduce

data_access = pd.read_csv("access.csv")
data_flint = pd.read_csv("flint.csv")
data_fqdn = pd.read_csv("fqdn.csv")
data_ip = pd.read_csv("ip.csv")
data_ipv6 = pd.read_csv("ipv6.csv")
data_label = pd.read_csv("label.csv")
# 在运行whois_json_csv和csv_merge函数后生成的两个文件
df = pd.read_csv('file_out.csv')
data_whois = pd.read_csv('whois.csv')


# whois转csv
def whois_json_csv():
    # 分别 读，创建文件
    json_fp = open("whois.json", "r", encoding='UTF-8')
    csv_fp = open("whois.csv", "w", encoding='UTF-8')
    # 提出表头和表的内容
    data_list = json.load(json_fp)
    sheet_title = data_list[0].keys()
    sheet_data = []
    for data in data_list:
        sheet_data.append(data.values())
    # 写入
    writer = csv.writer(csv_fp)
    writer.writerow(sheet_title)
    writer.writerows(sheet_data)
    # 关闭两个文件
    json_fp.close()
    csv_fp.close()
    # 去除空白行
    data_csv = pd.read_csv("whois.csv")
    # how=all删除的是全是缺失值的行，how=any删除存在有缺失值的行，inplace=true是在原基础上修改
    res = data_csv.dropna(how="all")
    res.to_csv("whois.csv", index=False)


# 最开始时合并csv文件access, fqdn, ip, ipv6, label
def csv_merge():
    # 对access表分割时间的具体数据
    data_access['time'] = pd.to_datetime(data_access['time'], format='%Y%m%d%H%M%S')
    data_access['year'] = data_access['time'].dt.year
    data_access['date'] = data_access['time'].dt.date
    data_access['month'] = data_access['time'].dt.month
    data_access['day'] = data_access['time'].dt.day
    data_access['hour'] = data_access['time'].dt.hour
    data_access['minute'] = data_access['time'].dt.minute

    # 合并
    data = pd.merge(data_fqdn, data_access, on='fqdn_no', how='left')
    # 合并ipv6和ipv4的数据为ip
    data_ip_a = pd.concat([data_ip, data_ipv6], sort=False)
    data = pd.merge(data, data_ip_a, on='encoded_ip', how='left')
    data = pd.merge(data, data_label, on='fqdn_no', how='left')

    data = data.fillna(-1)
    data = data[data['encoded_ip'] != -1]
    data.dropna(how='any', inplace=True)
    data.head()
    # 输出一共有多少行
    print(len(data['fqdn_no']))

    # 对域名的根域名以及对应 ip 的做切分操作，便于统计和交叉特征
    data['ip_tld'] = data['encoded_ip'].apply(lambda x: x.split('.')[0])
    data['encoded_fqdn_last'] = data['encoded_fqdn'].apply(lambda x: x.split('.')[-1])
    data['encoded_fqdn_first'] = data['encoded_fqdn'].apply(lambda x: x.split('.')[0])

    # 处理后，输出到csv文件
    data.to_csv('file_out.csv', encoding='utf-8')


# 特征工程

# 对hour，minute，day，latitude，longitude，count 数值特征统计最大、最小，均值，方差，中位数等特征
def feature_num():
    # hour
    hour_max = df.groupby(['fqdn_no'])['hour'].max()
    hour_min = df.groupby(['fqdn_no'])['hour'].min()
    hour_mean = df.groupby(['fqdn_no'])['hour'].mean()
    hour_var = df.groupby(['fqdn_no'])['hour'].var()
    hour_median = df.groupby(['fqdn_no'])['hour'].median()
    hour_std = df.groupby(['fqdn_no'])['hour'].std()
    # 转成字典类型,方便存入csv
    hour_max = pd.DataFrame({'hour_max': hour_max})
    hour_min = pd.DataFrame({'hour_min': hour_min})
    hour_mean = pd.DataFrame({'hour_mean': hour_mean})
    hour_var = pd.DataFrame({'hour_var': hour_var})
    hour_median = pd.DataFrame({'hour_median': hour_median})
    hour_std = pd.DataFrame({'hour_std': hour_std})

    # minute
    minute_max = df.groupby(['fqdn_no'])['minute'].max()
    minute_min = df.groupby(['fqdn_no'])['minute'].min()
    minute_mean = df.groupby(['fqdn_no'])['minute'].mean()
    minute_var = df.groupby(['fqdn_no'])['minute'].var()
    minute_median = df.groupby(['fqdn_no'])['minute'].median()
    minute_std = df.groupby(['fqdn_no'])['minute'].std()

    minute_max = pd.DataFrame({'minute_max': minute_max})
    minute_min = pd.DataFrame({'minute_min': minute_min})
    minute_mean = pd.DataFrame({'minute_mean': minute_mean})
    minute_var = pd.DataFrame({'minute_var': minute_var})
    minute_median = pd.DataFrame({'minute_median': minute_median})
    minute_std = pd.DataFrame({'minute_std': minute_std})

    # day
    day_max = df.groupby(['fqdn_no'])['day'].max()
    day_min = df.groupby(['fqdn_no'])['day'].min()
    day_mean = df.groupby(['fqdn_no'])['day'].mean()
    day_var = df.groupby(['fqdn_no'])['day'].var()
    day_median = df.groupby(['fqdn_no'])['day'].median()
    day_std = df.groupby(['fqdn_no'])['day'].std()

    day_max = pd.DataFrame({'day_max': day_max})
    day_min = pd.DataFrame({'day_min': day_min})
    day_mean = pd.DataFrame({'day_mean': day_mean})
    day_var = pd.DataFrame({'day_var': day_var})
    day_median = pd.DataFrame({'day_median': day_median})
    day_std = pd.DataFrame({'day_std': day_std})

    # latitude
    latitude_max = df.groupby(['fqdn_no'])['latitude'].max()
    latitude_min = df.groupby(['fqdn_no'])['latitude'].min()
    latitude_mean = df.groupby(['fqdn_no'])['latitude'].mean()
    latitude_var = df.groupby(['fqdn_no'])['latitude'].var()
    latitude_median = df.groupby(['fqdn_no'])['latitude'].median()
    latitude_std = df.groupby(['fqdn_no'])['latitude'].std()

    latitude_max = pd.DataFrame({'latitude_max': latitude_max})
    latitude_min = pd.DataFrame({'latitude_min': latitude_min})
    latitude_mean = pd.DataFrame({'latitude_mean': latitude_mean})
    latitude_var = pd.DataFrame({'latitude_var': latitude_var})
    latitude_median = pd.DataFrame({'latitude_median': latitude_median})
    latitude_std = pd.DataFrame({'latitude_std': latitude_std})

    # longitude
    longitude_max = df.groupby(['fqdn_no'])['longitude'].max()
    longitude_min = df.groupby(['fqdn_no'])['longitude'].min()
    longitude_mean = df.groupby(['fqdn_no'])['longitude'].mean()
    longitude_var = df.groupby(['fqdn_no'])['longitude'].var()
    longitude_median = df.groupby(['fqdn_no'])['longitude'].median()
    longitude_std = df.groupby(['fqdn_no'])['longitude'].std()

    longitude_max = pd.DataFrame({'longitude_max': longitude_max})
    longitude_min = pd.DataFrame({'longitude_min': longitude_min})
    longitude_mean = pd.DataFrame({'longitude_mean': longitude_mean})
    longitude_var = pd.DataFrame({'longitude_var': longitude_var})
    longitude_median = pd.DataFrame({'longitude_median': longitude_median})
    longitude_std = pd.DataFrame({'longitude_std': longitude_std})

    # count
    count_max = df.groupby(['fqdn_no'])['count'].max()
    count_min = df.groupby(['fqdn_no'])['count'].min()
    count_mean = df.groupby(['fqdn_no'])['count'].mean()
    count_var = df.groupby(['fqdn_no'])['count'].var()
    count_median = df.groupby(['fqdn_no'])['count'].median()
    count_std = df.groupby(['fqdn_no'])['count'].std()

    count_max = pd.DataFrame({'count_max': count_max})
    count_min = pd.DataFrame({'count_min': count_min})
    count_mean = pd.DataFrame({'count_mean': count_mean})
    count_var = pd.DataFrame({'count_var': count_var})
    count_median = pd.DataFrame({'count_median': count_median})
    count_std = pd.DataFrame({'count_std': count_std})

    # 合并
    regroup = [hour_max, hour_min, hour_mean, hour_var, hour_median, hour_std,
               minute_max, minute_min, minute_mean, minute_var, minute_median, minute_std,
               day_max, day_min, day_mean, day_var, day_median, day_std,
               latitude_max, latitude_min, latitude_mean, latitude_var, latitude_median, latitude_std, longitude_max,
               longitude_min, longitude_mean, longitude_var, longitude_median, longitude_std,
               count_max, count_min, count_mean, count_var, count_median, count_std]
    regroup = reduce(lambda left, right: pd.merge(left, right, how='outer', on="fqdn_no"), regroup)
    regroup.reset_index(inplace=True)
    regroup.to_csv('feature_data_num.csv')


#  对isp，encoded_ip，ip_tld，country，city，subdivision 做 nunique 和 count 特征
def feature_nuique():
    # nuique
    isp_nunique = df.groupby(['fqdn_no'])['isp'].nunique()
    eip_nunique = df.groupby(['fqdn_no'])['encoded_ip'].nunique()

    country_nunique = df.groupby(['fqdn_no'])['country'].nunique()
    city_nunique = df.groupby(['fqdn_no'])['city'].nunique()
    subdivision_nunique = df.groupby(['fqdn_no'])['subdivision'].nunique()

    isp_nunique = pd.DataFrame({'isp_unique': isp_nunique})
    eip_nunique = pd.DataFrame({'eip_unique': eip_nunique})

    country_nunique = pd.DataFrame({'country_unique': country_nunique})
    city_nunique = pd.DataFrame({'city_unique': city_nunique})
    subdivision_nunique = pd.DataFrame({'subdivision_unique': subdivision_nunique})

    # count
    isp_count = df.groupby(['fqdn_no'])['isp'].count()
    isp_count = pd.DataFrame({'isp_count': isp_count})

    # 合并
    regroup = [isp_nunique, eip_nunique, country_nunique, city_nunique, subdivision_nunique, isp_count]
    regroup = reduce(lambda left, right: pd.merge(left, right, how='outer', on="fqdn_no"), regroup)
    regroup.reset_index(inplace=True)
    regroup.to_csv('feature_data_unique.csv')


# 对encoded_fqdn 提取域名的一些文本特征
# 域名长度，出现字母次数，出现数字次数，单词长度占比，n级域名，特殊字符次数 #最大域名深度？
def feature_encoded_fqdn():
    domain = data_fqdn['encoded_fqdn']

    dns_Len = np.array(range(0, len(domain)))
    alpha_Cnt = np.array(range(0, len(domain)))
    digit_Cnt = np.array(range(0, len(domain)))
    word_Cnt = np.array(range(0, len(domain)))
    word_Len = np.array(range(0, len(domain)))
    word_Rate = np.zeros(len(domain))
    dot_Cnt = np.array(range(0, len(domain)))
    sp_Cnt = np.array(range(0, len(domain)))

    for i in range(len(domain)):
        # 初始化
        dns_len, alpha_cnt, digit_cnt, word_cnt, word_len, dot_cnt, sp_cnt = 0, 0, 0, 0, 0, 0, 0
        flag = False
        dns_len = len(domain[i])
        # 读取具体行列值，遍历分析
        for j in range(dns_len):
            if domain[i][j] == '[':
                word_cnt += 1
                flag = True
            elif domain[i][j] == ']':
                flag = False
            elif domain[i][j] == 'a':
                alpha_cnt += 1
                if flag:
                    word_len += 1
            elif domain[i][j] == '0':
                digit_cnt += 1
            elif domain[i][j] == '.':
                dot_cnt += 1
            elif not domain[i][j].isdigit() and not domain[i][j].isalpha():
                sp_cnt += 1

        # 写入数组
        dns_Len[i] = dns_len
        alpha_Cnt[i] = alpha_cnt
        digit_Cnt[i] = digit_cnt
        word_Cnt[i] = word_cnt
        word_Len[i] = word_len
        word_Rate[i] = word_len / dns_len
        dot_Cnt[i] = dot_cnt
        sp_Cnt[i] = sp_cnt

    dns_Len = pd.DataFrame(dns_Len, columns=['dns_len'])
    alpha_Cnt = pd.DataFrame(alpha_Cnt, columns=['alpha_cnt'])
    digit_Cnt = pd.DataFrame(digit_Cnt, columns=['digit_cnt'])
    word_Cnt = pd.DataFrame(word_Cnt, columns=['word_cnt'])
    word_Len = pd.DataFrame(word_Len, columns=['word_len'])
    word_Rate = pd.DataFrame(word_Rate, columns=['word_rate'])
    dot_Cnt = pd.DataFrame(dot_Cnt, columns=['dot_cnt'])
    sp_Cnt = pd.DataFrame(sp_Cnt, columns=['sp_cnt'])

    # 合并，axis=1为右连接列
    fqdn_no = data_fqdn['fqdn_no']
    regroup = [fqdn_no, dns_Len, alpha_Cnt, digit_Cnt, word_Cnt, word_Len, word_Rate, dot_Cnt, sp_Cnt]
    result = pd.concat(regroup, axis=1)
    result.to_csv('feature_encoded_fqdn.csv')


# ?没用上:计算信息熵
def feature_shannon_entropy(domain, lenth):
    label_counts = {}
    for featVec in domain:
        currentLabel = featVec[-1]
        if currentLabel not in label_counts.keys():
            label_counts[currentLabel] = 0
        label_counts[currentLabel] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / lenth
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


# 对hour, day, count,longitude,latitude做分位数特征(0.25和0.75)，刻画不同时段的变化情况     ## 简单的来说就是0.25或0.75的总样本数大于某个值,quantile()就是求这个值
def feature_quantile():
    hour_quantile_25 = df.groupby(['fqdn_no'])['hour'].quantile(0.25)
    hour_quantile_75 = df.groupby(['fqdn_no'])['hour'].quantile(0.75)
    hour_quantile_25 = pd.DataFrame({'hour_quantile_25': hour_quantile_25})
    hour_quantile_75 = pd.DataFrame({'hour_quantile_75': hour_quantile_75})

    day_quantile_25 = df.groupby(['fqdn_no'])['day'].quantile(0.25)
    day_quantile_75 = df.groupby(['fqdn_no'])['day'].quantile(0.75)
    day_quantile_25 = pd.DataFrame({'day_quantile_25': day_quantile_25})
    day_quantile_75 = pd.DataFrame({'day_quantile_75': day_quantile_75})

    count_quantile_25 = df.groupby(['fqdn_no'])['count'].quantile(0.25)
    count_quantile_75 = df.groupby(['fqdn_no'])['count'].quantile(0.75)
    count_quantile_25 = pd.DataFrame({'count_quantile_25': count_quantile_25})
    count_quantile_75 = pd.DataFrame({'count_quantile_75': count_quantile_75})

    longitude_quantile_25 = df.groupby(['fqdn_no'])['longitude'].quantile(0.25)
    longitude_quantile_75 = df.groupby(['fqdn_no'])['longitude'].quantile(0.75)
    latitude_quantile_25 = df.groupby(['fqdn_no'])['latitude'].quantile(0.25)
    latitude_quantile_75 = df.groupby(['fqdn_no'])['latitude'].quantile(0.75)

    longtide_quantile_25 = pd.DataFrame({'longitude_quantile_25': longitude_quantile_25})
    longtide_quantile_75 = pd.DataFrame({'longitude_quantile_75': longitude_quantile_75})
    latitude_quantile_25 = pd.DataFrame({'latitude_quantile_25': latitude_quantile_25})
    latitude_quantile_75 = pd.DataFrame({'latitude_quantile_75': latitude_quantile_75})

    regroup = [hour_quantile_25, hour_quantile_75, day_quantile_25, day_quantile_75,
               count_quantile_25, count_quantile_75, longtide_quantile_25, longtide_quantile_75,
               latitude_quantile_25, latitude_quantile_75]

    regroup = reduce(lambda left, right: pd.merge(left, right, how='outer', on="fqdn_no"), regroup)
    regroup.reset_index(inplace=True)
    regroup.to_csv('feature_quantile.csv')


# 对whois.csv提取特征: whois记录数量，最早创建日期，最晚创建日期，更新次数，最早更新，最晚更新，注册邮箱数，注册国家数
def feature_whois():
    record_cnt = data_whois.groupby(['fqdn_no'])['nameservers'].count()
    createddate_min = data_whois.groupby(['fqdn_no'])['createddate'].min()
    expiresdate_max = data_whois.groupby(['fqdn_no'])['expiresdate'].max()
    update_cnt = data_whois.groupby(['fqdn_no'])['updateddate'].nunique()
    update_min = data_whois.groupby(['fqdn_no'])['updateddate'].min()
    update_max = data_whois.groupby(['fqdn_no'])['updateddate'].max()
    registrantemail_cnt = data_whois.groupby(['fqdn_no'])['registrant_email'].nunique()
    registrantcountry_cnt = data_whois.groupby(['fqdn_no'])['registrant_country'].nunique()

    record_cnt = pd.DataFrame({'record_cnt': record_cnt})
    createddate_min = pd.DataFrame({'createddate_min': createddate_min})
    expiresdate_max = pd.DataFrame({'expiresdate_max': expiresdate_max})
    update_cnt = pd.DataFrame({'update_cnt': update_cnt})
    update_min = pd.DataFrame({'update_min': update_min})
    update_max = pd.DataFrame({'update_max': update_max})
    registrantemail_cnt = pd.DataFrame({'registrantemail_cnt': registrantemail_cnt})
    registrantcountry_cnt = pd.DataFrame({'registrantcounty_cnt': registrantcountry_cnt})

    regroup = [record_cnt, createddate_min, expiresdate_max, update_cnt, update_min, update_max, registrantemail_cnt,
               registrantcountry_cnt]
    regroup = reduce(lambda left, right: pd.merge(left, right, how='outer', on="fqdn_no"), regroup)
    regroup.reset_index(inplace=True)
    regroup.to_csv('feature_whois.csv')


# 对flint.csv: dns解析数，dns解析类，最早解析，最晚解析，每天请求次数
def feature_flint():
    flint_dns_cnt = data_flint.groupby(['fqdn_no'])['date'].count()
    flint_dns_type = data_flint.groupby(['fqdn_no'])['flintType'].nunique()
    flint_date_min = data_flint.groupby(['fqdn_no'])['date'].min()
    flint_date_max = data_flint.groupby(['fqdn_no'])['date'].max()
    flint_req_cnt = data_flint.groupby(['fqdn_no'])['requestCnt'].count()

    flint_dns_cnt = pd.DataFrame({'flint_dns_cnt': flint_dns_cnt})
    flint_dns_type = pd.DataFrame({'flint_dns_type': flint_dns_type})
    flint_date_min = pd.DataFrame({'flint_date_min': flint_date_min})
    flint_date_max = pd.DataFrame({'flint_date_max': flint_date_max})
    flint_req_cnt = pd.DataFrame({'flint_req_cnt': flint_req_cnt})

    regroup = [flint_dns_cnt, flint_dns_type, flint_date_min, flint_date_max, flint_req_cnt]
    regroup = reduce(lambda left, right: pd.merge(left, right, how='outer', on="fqdn_no"), regroup)
    regroup.reset_index(inplace=True)
    regroup.to_csv('feature_flint.csv')


# 提取每个不同时间段、经纬度的访问数,做value_counts()特征
def feature_per_access():
    # 按fqdn_no 对hour做values.counts()特征并升序排序     #是否选取比率 normalize=True？
    group_hour = df.groupby(['fqdn_no', 'hour'])['count'].sum()
    group_hour = pd.DataFrame({'cnt': group_hour})
    group_hour.reset_index(inplace=True)
    group_hour.to_csv('rough/valuecounts_hour.csv')
    # date
    group_date = df.groupby(['fqdn_no', 'date'])['count'].sum()
    group_date = pd.DataFrame({'cnt': group_date})
    group_date.reset_index(inplace=True)
    group_date.to_csv('rough/valuecounts_date.csv')

# 针对不同时间段的value_counts()特征，做一个行列变换函数来处理得到的数据 ?
def change_rowandcol(data, column2, column3):
    # data = pd.read_csv('test.csv', low_memory=False)
    domain = data['fqdn_no']
    domain_hour = data[column2]
    domain_hourcnt = data[column3]
    domain_dup = domain.drop_duplicates()
    group = pd.DataFrame({'fqdn_no': domain_dup}, index=None)
    group.reset_index(inplace=True)

    # 需要的长度和字典大小
    new_columns = []
    colum_num = data[column2].unique()
    flag = False

    for i in colum_num:
        for j in range(len(domain) - 1):
            if domain_hour[j] == i and domain[j] == domain[j + 1]:
                new_columns.append(domain_hourcnt[j])
                flag = True
            elif domain_hour[j] == i and domain[j] != domain[j + 1]:
                new_columns.append(domain_hourcnt[j])
                flag = False
            elif flag == False and domain[j] != domain[j + 1]:
                new_columns.append(0)
            elif flag == True and domain[j] != domain[j + 1]:
                flag = False
        if domain_hour[len(domain) - 1] == i:
            new_columns.append(domain_hourcnt[len(domain) - 1])
        elif flag == False:
            new_columns.append(0)
        new_df = pd.DataFrame(new_columns)
        group = pd.concat([group, new_df], axis=1)
        new_columns = []

    # 重命名列为原来的column2+数字,删去index列,输出
    col = ['index', 'fqdn_no']
    for k in range(data[column2].nunique()):
        col.append(column2 + str(k))
    group.columns = col
    group = group.drop(['index'], axis=1)
    return group

def set_label_0():
    return 0

def set_label_1():
    return 1

def get_label(data):
    if data['family_no'] == -1:
        return 0
    else:
        return 1


def merge_features():
    label = pd.read_csv('data/label.csv')
    tmp = []
    for row in label.iterrows():  # 遍历数据表,填入-1
        tmp.append(-1)
    label['tmp'] = tmp

    feature_data_num = pd.read_csv('feature_data_num.csv', index_col=0)
    feature_data_unique = pd.read_csv('feature_data_unique.csv', index_col=0)
    feature_quantile = pd.read_csv('feature_quantile.csv', index_col=0)
    feature_encoded_fqdn = pd.read_csv('feature_encoded_fqdn.csv', index_col=0)
    feature_flint = pd.read_csv('feature_flint.csv', index_col=0)
    feature_whois = pd.read_csv('feature_whois.csv', index_col=0)

    feature_dif_date = pd.read_csv('feature_dif_date.csv', index_col=0)
    feature_dif_hour = pd.read_csv('feature_dif_hour.csv', index_col=0)

    data = pd.merge(feature_data_num, feature_data_unique, on='fqdn_no', how='left')
    data = pd.merge(data, feature_quantile, on='fqdn_no', how='left')
    data = pd.merge(data, feature_encoded_fqdn, on='fqdn_no', how='left')

    data = pd.merge(data, feature_flint, on='fqdn_no', how='left')
    data = pd.merge(data, feature_whois, on='fqdn_no', how='left')

    data = pd.merge(data, feature_dif_date, on='fqdn_no', how='left')
    data = pd.merge(data, feature_dif_hour, on='fqdn_no', how='left')

    # 输出全体特征
    data.to_csv('feature_all.csv')

    # 取得有标签的数据data_labeled，并对已经分号的恶意域名加一列用来标记他们
    data_labeled = pd.merge(data, label, on='fqdn_no', how='inner')

    # 取得全体数据(包括标签的)
    data = pd.merge(data, label, on='fqdn_no', how='left')
    domain_tmp = data['tmp']
    domain_tmp.fillna(0, inplace=True)

    domain_no = data['family_no']
    domain_no.fillna(-1, inplace=True)

    # 原始多家族的恶意域名
    data_labeled.to_csv('labeled.csv')
    # 所有家族视为1类，family_no全部置1
    data_labeled['family_no'] = data_labeled.apply(set_label_1, axis=1)
    data_labeled.to_csv('labeled_1.csv')

    # 获得恶意域名和白域名的所有集合，恶意域名为1
    data['family_no'] = data.apply(get_label, axis=1)
    data.to_csv('labeled_all.csv')

    # 去掉标签的数据(视为0)
    data = data[data['tmp'] != -1]
    data['family_no'] = data.apply(set_label_0, axis=1)
    data.reset_index(inplace=True)
    data = data.drop(['index'], axis=1)
    data.to_csv('unlabeled_test.csv')

if __name__ == "__main__":

    data = pd.read_csv('rough/valuecounts_date.csv')
    column2 = 'date'
    column3 = 'cnt'
    group = change_rowandcol(data, column2, column3)
    group.to_csv('feature_dif_date.csv')
    merge_features()
