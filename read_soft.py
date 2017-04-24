import GEOparse
import pandas as pd
import numpy as np
import sys
import pickle

def ReadGPL(name):
    # gse = GEOparse.get_GEO(geo = name, destdir = 'data/')
    # gse = GEOparse.get_GEO(geo = name, destdir = 'data/')
    PRINT_GPL_CONTENT = 0
    gse = GEOparse.get_GEO(filepath = name)
    if (PRINT_GPL_CONTENT == 1):
        for gpl_name, gpl in gse.gpls.items():
            print("Name: ", gpl_name)
            print("Metadata:",)
            for key, value in gpl.metadata.items():
                print(" - %s : %s" % (key, ", ".join(value)))
            print("Table data:",)
            print(gpl.table.head())
            break
        sys.exit()
    else:
        ids = []
        gene_ids = []
        for gpl_name, gpl in gse.gpls.items():
            if (ids == []):
                gene_ids = np.array(gpl.table['Gene Symbol'])
                ids = np.array(gpl.table['ID'])
            else:
                # if multiple GPLs in one GSE
                gene_ids = np.concatenate((gene_ids, np.array(gpl.table['Gene Symbol'])), axis = 0)
                ids = np.concatenate((ids, np.array(gpl.table['ID'])), axis = 0)
        dict_id_gene = dict(zip(ids,gene_ids))
        # print(dict_id_gene)
        # print(type(dict_id_gene))
        # print(dict_id_gene)
    return dict_id_gene


def ReadGDS(file_name, y):
    gse = GEOparse.get_GEO(filepath = file_name)
    raw_data = gse.table
    # raw_data.convert_objects(convert_numeric=True)
    data_labels = raw_data.columns[2:2+len(y)]
    data = []
    for label in data_labels:
        data.append(raw_data.loc[:,label])
    gene_ids = raw_data.loc[:,raw_data.columns[1]]
    # print(type(data))
    return (np.array(gene_ids), np.array(data), np.array(y))

def ReadGSE(file_name):
    gse = GEOparse.get_GEO(filepath = file_name)
    PRINT_GEO_CONTENT = 0
    if (PRINT_GEO_CONTENT == 1) :
        for gsm_name, gsm in gse.gsms.items():
            print('Name', gsm_name)
            print('Meta data')
            for key, value in gsm.metadata.items():
                print(" - %s : %s" % (key, ", ".join(value)))
            # print ("Table data:")
            # print (gsm.columns)
            print (gsm.table)

    labels = []
    data = []
    # traverse all gsms
    for gsm_name, gsm in gse.gsms.items():
        for key, value in gsm.metadata.items():
            if (key == 'title'):
                labels.append(value[0])
                # print(value)
        gene_ids = gsm.table['ID_REF'].values
        tmp_data = gsm.table['VALUE'].values
        data.append(tmp_data)
        # data = np.concatenate((data,tmp_data),axis = 0)
    # sys.exit()
    y = []
    # ND 0, RA 1, OA 2
    for label in labels:
        if ('ND' in label or 'Normal' in label):
            y.append(0)
        if ('RA' in label or 'Rheumatoid' in label):
            y.append(1)
        if ('OA' in label or 'Ost' in label):
            y.append(2)
    print(file_name)
    check_y = np.array(y)
    print('ND has {}, RA has {}, OA has {}'.format(np.sum(check_y==0), np.sum(check_y==1), np.sum(check_y==2)))
    return (np.array(gene_ids), np.array(data), np.array(y))

def filter_top_batch(id_list, org_list, N):
    arg_list = org_list.argsort()[-N:][::-1]
    diff_value = org_list[arg_list]
    diff_ids = id_list[arg_list]
    # threshold = np.percentile(np.abs(org_list), percentage)
    # selected_list = (np.abs(org_list) > threshold) * org_list
    # diff_value = selected_list[selected_list!=0]
    # diff_ids = id_list[selected_list!=0]
    # print(diff_value)
    # print(diff_ids)
    return (diff_ids, diff_value, arg_list)

def substring(small_string, large_string):
    if (small_string in large_string):
        return True
    else:
        return False

def DataWash(data, ids, y, GDS_ONLY, N_FEATURES = 500):
    print(80*'-')
    print('In data wash')

    # deal with GDS data
    for i in range(0,3):
        data[i] = data[i].astype(float)
        if (i == 0):
            gds_data = data[i]
            gds_y = y[i]
        else:
            gds_data = np.concatenate((gds_data, data[i]), axis = 0)
            gds_y = np.concatenate((gds_y, y[i]), axis = 0)
    print('Three GDS files: data {}, y {}, id {}'.format(
        np.shape(gds_data),
        np.shape(gds_y),
        np.shape(ids[0])))
    gds_picked_ids, gds_picked_arg = feature_extract_GDS(gds_data, ids[0], gds_y, N_FEATURES)
    # These are additonal data
    # data_list = []
    # y_list = []
    gse_id_list = []
    for i in range(3,len(data)):
        # try match picked ids
        cnt = 0
        tmp = []
        if (i == 4):
            parent_list = ids[i][0:22283]
        else:
            parent_list = ids[i]
        for pid in gds_picked_ids:
            if (substring(pid,parent_list)):
                tmp.append(pid)
            else:
                cnt = cnt + 1
        gse_id_list.append(tmp)
        print('{} ids do not exits in id list {}, total picked ids are {}'.format(cnt, i, len(tmp)))
    # common ids
    for i  in range(3-3,len(data)-3):
        if (i == 0):
            gse_common_ids = gse_id_list[0]
        else:
            # gse2 has mixed data length, lets normalize them here
            gse_common_ids = list(set(gse_common_ids).intersection(gse_id_list[i]))
    print('{} common ids are found'.format(len(gse_common_ids)))
    print(gse_common_ids)

    # fetch data and y labels for these common ids

    # common_ids's arg in gds
    fetching_gds_args = find_args(gse_common_ids, ids[0])
    merge_data = gds_data[:,fetching_gds_args]
    merge_y = gds_y

    # append gse data
    for i in range(3, len(data)):
        for elem in data[i]:
            fetching_dse_args = find_args(gse_common_ids, np.array(ids[i]))
            tmp = np.array([elem[fetching_dse_args]])
            merge_data = np.concatenate((merge_data,tmp),axis = 0)
        # print(np.shape(y[i]))
        merge_y = np.concatenate((merge_y, y[i]), axis = 0)
    # pick the genes and also their indexes

    if (GDS_ONLY == 1):
        picked_arg = gds_picked_arg
        picked_ids = gds_picked_ids
        out_y = gds_y
        out_dat = gds_data[:,picked_arg]
    else:
        picked_ids = gse_common_ids
        out_y = merge_y
        out_dat = merge_data


    print(np.shape(out_dat))
    print(np.shape(out_y))


    # deal with GSE data
    # for i in range(3,len(data)):
    #     data[i] = data[i].astype(float)
    #     feature_extract(data[i], ids[i], y[i])
    return (picked_ids, out_dat, out_y)

def find_args(selected_ids, full_id_list):
    arg_list = []
    id_list = full_id_list.tolist()
    for sel_id in selected_ids:
        arg_list.append(id_list.index(sel_id))
    return arg_list

def feature_extract_GDS(data, ids ,y, n):
    # ND 0, RA 1, OA 2
    data_nc = []
    data_ra = []
    data_oa = []
    for i in range(0,len(y)):
        if (y[i] == 0):
            data_nc.append(data[i,:])
        if (y[i] == 1):
            data_ra.append(data[i,:])
        if (y[i] == 2):
            data_oa.append(data[i,:])
    if (data_nc != []):
        data_nc = np.array(data_nc)
        # print(np.shape(data_nc))
        nc_mean = np.mean(data_nc, axis = 0)
    if (data_ra != [] and data_nc != []):
        data_ra = np.array(data_ra)
        ra_mean = np.mean(data_ra, axis = 0)
        ra_mean_diff = np.abs(nc_mean - ra_mean)
        # print(ra_mean_diff)
    if (data_oa != [] and data_nc != []):
        data_oa = np.array(data_oa)
        oa_mean = np.mean(data_oa, axis = 0)
        oa_mean_diff = np.abs(nc_mean - oa_mean)
    if (data_ra != [] and data_oa != []):
        oa_ra_mean_diff = np.abs(oa_mean - ra_mean)

    if ('oa_mean_diff' in locals() and
        'ra_mean_diff' in locals() and
        'oa_ra_mean_diff' in locals()):
        mean_diff_sum = oa_mean_diff + ra_mean_diff + oa_ra_mean_diff
        selected_ids, selected_values, arg_list = filter_top_batch(ids, mean_diff_sum, n)
        # print(selected_ids)
        # print(selected_values)
    return (selected_ids, arg_list)

def PreprocessData(GDS_ONLY = 1, N_features = 50):
    # yg values are determined from the .soft file
    # ND 0, RA 1, OA 2
    RAW_DATA_EXISTS = 1
    if (RAW_DATA_EXISTS == 0):
        yg1 = 10 * [0] + 10 * [2] + 10 * [1]
        g1_id, g1_table, g1_y= ReadGDS('data/GDS5401_full.soft', yg1)
        yg2 = 10 * [1] + 6 * [2]
        g2_id, g2_table, g2_y= ReadGDS('data/GDS5402_full.soft', yg2)
        yg3 = 10 * [0] + 13 * [1] + 10 * [2]
        g3_id, g3_table, g3_y= ReadGDS('data/GDS5403_full.soft', yg3)

        gse1_id_keys, gse1_data, gse1_y = ReadGSE('data/GSE1919_family.soft.gz')
        gse2_id_keys, gse2_data, gse2_y = ReadGSE('data/GSE12021_family.soft.gz')
        # index = 0
        # for elem in gse2_data:
        #     tmp[index,:] = elem[:]
        #     index += 1
        # print(np.shape(gse1_data))
        # print(np.shape(gse1_data[0]))
        # print(np.shape(np.array(gse2_data)))
        # print(np.shape(tmp))
        # print(np.shape(gse2_data[0]))
        # print(type(gse1_data))
        # print(type(gse1_data[0]))
        # print(type(gse2_data))
        # print(type(gse2_data[0]))
        # index = 0
        # for elem in gse2_data:
        #     gse2_data[index,:] = elem[0:22283]
        #     print(len(elem))
        # sys.exit()
        gse3_id_keys, gse3_data, gse3_y = ReadGSE('data/GSE48780_family.soft.gz')
        gse1_id_dict = ReadGPL('data/GSE1919_family.soft.gz')
        gse2_id_dict = ReadGPL('data/GSE12021_family.soft.gz')
        gse3_id_dict = ReadGPL('data/GSE48780_family.soft.gz')

        gse1_id = [gse1_id_dict[i] for i in gse1_id_keys]
        gse2_id = [gse2_id_dict[i] for i in gse2_id_keys]
        # print(np.shape(gse3_id_keys))
        # print(np.shape(gse3_id_dict))
        gse3_id = [gse3_id_dict[i] for i in gse3_id_keys]

        # filter out unexpected text
        g1 = g1_table
        g1[g1=='null'] = 0
        g2 = g2_table
        g2[g2=='null'] = 0
        g3 = g3_table
        g3[g3=='null'] = 0

        data_agg = [g1, g2, g3, gse1_data, gse2_data, gse3_data]
        id_agg = [g1_id, g2_id, g3_id, gse1_id, gse2_id, gse3_id]
        y_agg = [g1_y, g2_y, g3_y, gse1_y, gse2_y, gse3_y]
        print('shape of y is {}, {}, {}'.format(gse1_y, gse2_y, gse3_y))
        print('shape of y is {}, {}, {}'.format(np.shape(gse1_y), np.shape(gse2_y), np.shape(gse3_y)))
        with open('data/raw_data.pkl', 'wb') as f:
            pickle.dump((data_agg, id_agg, y_agg),f)
        print('Finished saving data')
        sys.exit()
    else:
        with open('data/raw_data.pkl', 'rb') as f:
            data_agg, id_agg, y_agg = pickle.load(f)

    # Perform data wash, select some genes with large variances between controls
    # and infected groups

    sel_ids, sel_data, sel_y= DataWash(data_agg, id_agg, y_agg, GDS_ONLY, N_features)
    return (sel_ids, sel_data, sel_y)

if __name__ == '__main__':
    PreprocessData()
