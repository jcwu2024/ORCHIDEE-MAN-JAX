#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import subprocess
import math

import random
from time import time
import numpy as np
import math


import os, re, sys, glob
from datetime import datetime
import pandas as pd
import time
import netCDF4 as nc


#(1)参数需要增加
#(2)传入参数为读取的值

def func_replace(filein_func, name_func, idx_func, value_func):
    import re
    with open(filein_func, 'r') as fin_func:
        lines_func = fin_func.readlines()
        flag = 0
        for i_func, line_func in enumerate(lines_func):
            if re.match(name_func, line_func):
                lines_func[i_func] = line_func[:idx_func] + str(value_func) +'\n'
                flag += 1
        if flag==0:
            #print('Cannot find %s'%name_func)
            print 'Cannot find'
    fin_func.close()
    with open(filein_func, 'w') as fin_func:
        fin_func.writelines(lines_func)
    fin_func.close()

def extract_file(Igrid,d_outloc,iterid,simid, Iage,arg1,arg2,arg3,arg4,arg5):

    d_outloc_grid = d_outloc+Igrid+'/I'+str(iterid)+'/S'+str(simid)+'_'+str(arg2)[:6]+'_'+str(arg3)[:6]+'_'+str(arg4)[:6]+'_'+str(arg5)[:6]+'/'

    pt_nc=d_outloc_grid+'stomate_history_'+str(int(1961+Iage-1))+'.nc'

    #print 'out:d_outloc_grid'
    #print(d_outloc_grid)
    #print 'out:pt_nc'
    #print(pt_nc)

    if os.path.exists(pt_nc):

        dt_nc = nc.Dataset(pt_nc)
        dt_l   = dt_nc['LEAF_M'][0,13,0,0]
        dt_s_a = dt_nc['SAP_M_AB'][0,13,0,0]
        dt_h_a = dt_nc['HEART_M_AB'][0,13,0,0]
        dt_agr_s_st = dt_nc['AGR_SAP_ST_M'][0,13,0,0]
        dt_agr_h_st = dt_nc['AGR_HRT_ST_M'][0,13,0,0]
        dt_agr_s_pn = dt_nc['AGR_SAP_PN_M'][0,13,0,0]
        dt_agr_h_pn = dt_nc['AGR_HRT_PN_M'][0,13,0,0]

        dt_s_b = dt_nc['SAP_M_BE'][0,13,0,0]
        dt_h_b = dt_nc['HEART_M_BE'][0,13,0,0]
        dt_r   = dt_nc['ROOT_M'][0,13,0,0]

        dt_GPP   = dt_nc['GPP'][0,13,0,0]
        dt_NPP   = dt_nc['NPP'][0,13,0,0]

        dt_AGB=dt_l+dt_s_a+dt_h_a+dt_agr_s_st+dt_agr_h_st+dt_agr_s_pn+dt_agr_h_pn
        dt_BGB=dt_s_b+dt_h_b+dt_r

        return dt_AGB,dt_BGB,dt_GPP,dt_NPP

    else:
        return -9999,-9999,-9999,-9999


def mod_extract(Igrid,d_outloc,Itag, iterid,simid, arg1,arg2,arg3,arg4,arg5):
    
    p_std_AGB=[0, 1]
    p_std_BGB=[0, 1]
    p_std_GPP=[0, 1]
    p_std_NPP=[0, 1]

    #此处为没有归一化的观测数据列表
    #dt_obs=pd.read_csv('/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_240430/dt_obs.csv')    
    #dt_obs=pd.read_csv('/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_240430/dt_obs_gf_extglo.csv')    
    dt_obs=pd.read_csv('/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_250919/dt_obs_gf_extglo.csv')    
    dt_obs_grid=dt_obs[dt_obs.Igrid==Igrid]
    dt_obs_grid=dt_obs_grid.reset_index(drop=True)
   
    dt_obs_grid['AGB_model']=np.nan
    dt_obs_grid['BGB_model']=np.nan
    dt_obs_grid['GPP_model']=np.nan
    dt_obs_grid['NPP_model']=np.nan
    
    for I in range(len(dt_obs_grid)):
        
        #print('extract begin')
        res_extract=extract_file(Igrid,d_outloc,iterid,simid, dt_obs_grid.age[I],arg1,arg2,arg3,arg4,arg5)
        #print('extract end')
        #print(res_extract)
        
        dt_obs_grid.AGB_model[I]=res_extract[0]
        dt_obs_grid.BGB_model[I]=res_extract[1]
        dt_obs_grid.GPP_model[I]=res_extract[2]
        dt_obs_grid.NPP_model[I]=res_extract[3]
        
    dt_obs_grid.AGB_model=(dt_obs_grid.AGB_model*0.02 - p_std_AGB[0]) / p_std_AGB[1] #全部用归一化结果
    dt_obs_grid.BGB_model=(dt_obs_grid.BGB_model*0.02 - p_std_BGB[0]) / p_std_BGB[1]
    dt_obs_grid.GPP_model=(dt_obs_grid.GPP_model - p_std_GPP[0]) / p_std_GPP[1]
    dt_obs_grid.NPP_model=(dt_obs_grid.NPP_model - p_std_NPP[0]) / p_std_NPP[1]
   
    d_modelout='/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_250919/modelout_sen/'+Itag+'/'
    if not os.path.exists(d_modelout):
        os.makedirs(d_modelout)
    dt_obs_grid.to_csv(d_modelout+'/modelout_'+Igrid+'.csv')



# In[ ]:


def make_job(d_ws,d_outloc,Itag, Igrid,iterid,simid, arg1,arg2,arg3,arg4,arg5, arg_b1,arg_b2,arg_b3,arg_b4,arg_b5,arg_b7):
    

    d_grid_job = d_ws+'/sen/'+Itag+'/Job/'+Igrid+'/I'+str(iterid)+'/S'+str(simid)+'_'+str(arg2)[:6]+'_'+str(arg3)[:6]+'_'+str(arg4)[:6]+'_'+str(arg5)[:6]+'/'
    if not os.path.exists(d_grid_job):
        os.makedirs(d_grid_job)

        
    #(1)复制并修改run.def
    rundname = 'run.def_'+str(arg2)[:6]+'_'+str(arg3)[:6]+'_'+str(arg4)[:6]+'_'+str(arg5)[:6]
    pt_rund = d_grid_job + rundname
    txt_cp1 = 'cp -r run.def.vn '+pt_rund
    print txt_cp1
    os.system(txt_cp1)

    p_arjv = 100/float(arg2) + 25*0.035
    arg_b6=1-arg_b5
 
    #func_replace(pt_rund, 'ARJV__00014'   , 12 , 2.59)
    func_replace(pt_rund, 'ARJV__00014'   , 12 , p_arjv)
    func_replace(pt_rund, 'SLA__00014'    , 11 , arg1)
    func_replace(pt_rund, 'SLA_MAX__00014', 15 , arg1)
    func_replace(pt_rund, 'SLA_MIN__00014', 15 , arg1)

    func_replace(pt_rund, 'VCMAX25__00014', 15 , arg2)
    func_replace(pt_rund, 'MAINT_RESP_SLOPE_C__00014', 26 , arg3)
    func_replace(pt_rund, 'ALLOC_MIN__00014',          17 , arg4)
    func_replace(pt_rund, 'RESIDENCE_TIME__00014'    , 22 , arg5)

    func_replace(pt_rund, 'CONTROL_SALINITY_MIN', 21 , arg_b1)
    func_replace(pt_rund, 'CONTROL_INUDATE_MIN', 20 , arg_b2)
    func_replace(pt_rund, 'AGB_AGR_VEN_ALL_ST', 19 , arg_b3)
    func_replace(pt_rund, 'AGB_AGR_VEN_ALL_PN', 19 , arg_b4)
    func_replace(pt_rund, 'WEIGHT_VEN_A', 13 , arg_b5)
    func_replace(pt_rund, 'WEIGHT_VEN_B', 13 , arg_b6)
    func_replace(pt_rund, 'DEN_MAN', 8 , arg_b7)

    #(2)复制并修改job文件(经纬度,输出路径)
        
    lonG=float(Igrid[0:5])-180  
    latG=float(Igrid[6:11])-90 
    
    latS, latN = latG-1, latG+1
    lonW, lonE = lonG-1, lonG+1

    jobname = 'Job_'+Igrid
    pt_job = d_grid_job + jobname + '.sh' 

    d_outloc_job=d_outloc+Igrid+'/I'+str(iterid)+'/S'+str(simid)+'_'+str(arg2)[:6]+'_'+str(arg3)[:6]+'_'+str(arg4)[:6]+'_'+str(arg5)[:6]+'/'

    
    txt_cp2 = 'cp -r Job0_bio '+pt_job
    print txt_cp2
    os.system(txt_cp2)


    items_run = ['west_bound','east_bound','north_bound','south_bound','OUTLOC','RUNDEF']
    nitem_run = len(items_run)
    idxs_run  = [11,11,12,12,7,7]
    values_run = [lonW,lonE,latN,latS,d_outloc_job,pt_rund]

    for j in range(nitem_run):
        func_replace(pt_job, items_run[j], idxs_run[j], values_run[j])

    
    return pt_job 
    
    

def make_group(d_ws,filelist,Itag):
    nproc = 32
    nproc2= 32

    filelist = np.sort(filelist)

    nsite = len(filelist)
    #----divide into groups based on nproc----------
    ngroup = int((nsite+nproc2-1)/nproc2)
    d_grpjob = d_ws + 'sen/'+Itag+'/Group_job/'
    if not os.path.exists(d_grpjob):
        os.makedirs(d_grpjob)

    #------create group job---------
    for i in np.arange(ngroup):
        Gjobname = 'G' + str(i)
        groupfile = d_grpjob + Gjobname + '.sh'
        #os.system('cp %sjob_prl.sh %s'%(d_ws, groupfile))
        os.system('cp %ssub_temp_df.sh %s'%(d_ws, groupfile))
        ista = i*nproc2
        if i < ngroup-1:
            iend = (i+1)*nproc2
            nproc_tmp = nproc2
        else:
            iend = nsite
            nproc_tmp = nsite - i*nproc2

        with open(groupfile,'a') as f:
            for job in filelist[ista:iend]:
                line_tmp  = 'srun -n 1 --exclusive ' + job + ' &\n'
                f.writelines(line_tmp)
            f.writelines('wait')
        f.close()

    #txt_chmod='chmod 777 '+d_alljob+'*'
    #os.system(txt_chmod)
    txt_chmod='chmod 777 '+d_grpjob+'*'
    os.system(txt_chmod)

    #for i in np.arange(ngroup):
    #    Gjobname = 'G' + str(i)
    #    groupfile = d_grpjob + Gjobname + '.sh'
    #    #txt_sub='qsub '+groupfile
    #    #txt_sub='bsub -q q_x86_share_1 -n '+str(nproc)+' '+groupfile
    #    txt_sub='bsub -q q_x86_share -n 1 '+' '+groupfile


# In[ ]:


if __name__ == "__main__":

    d_ws='/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_250919/'
    
    #路径和参数根据敏感性分析百分比调整
    #Job和Group的路径也要修改

    tab_res_cali=pd.read_csv(d_ws+'tab_res_cali.csv',dtype='str')
    
    ls_para=['arg2','arg3','arg4','arg5', 'arg_b1','arg_b2','arg_b3','arg_b4','arg_b5','arg_b7']
    ls_scale_factor=[0.7,0.8,0.9,1.0,1.1,1.2,1.3]


    #for Ipara in range(len(ls_para)):
    #    #for Isf in ls_scale_factor:
    #    for Isf in [0.7,0.8,0.9,1.1,1.2,1.3]:
    for Ipara in [0]:
        for Isf in [1.0]:

            #用作调节10个参数的系数
            scale_para_sen=[1,1,1,1,1,1,1,1,1,1]
            scale_para_sen[Ipara]=scale_para_sen[Ipara]*Isf

            Itag=ls_para[Ipara]+'_'+str(Isf)

            d_outloc='/public/share/qhcess/qhcess2/User/zzhao/OUT/orc_calibrate_250919_sen/'+Itag+'/'
               
            #print(d_outloc)
            #1)先提交1），制作脚本
            '''
            filelist=[]
            for I in range(len(tab_res_cali)):
                iterid=tab_res_cali.Iite[I]
                simid=tab_res_cali.Isim[I]
                arg1=0.0153
                arg2=float(tab_res_cali.p2[I]) * scale_para_sen[0]
                arg3=float(tab_res_cali.p3[I]) * scale_para_sen[1]
                arg4=float(tab_res_cali.p4[I]) * scale_para_sen[2]
                arg5=float(tab_res_cali.p5[I]) * scale_para_sen[3]
                arg_b1=0.5 * scale_para_sen[4]
                arg_b2=0.5 * scale_para_sen[5]
                arg_b3=200 * scale_para_sen[6]
                arg_b4=40 * scale_para_sen[7]
                arg_b5=0.5 * scale_para_sen[8]
                #arg_b6=0.6 * scale_para_sen[9]
                arg_b7=1000 * scale_para_sen[9]
        
                pt_job=make_job(d_ws,d_outloc,Itag, tab_res_cali.Igrid[I], iterid,simid, arg1,arg2,arg3,arg4,arg5, arg_b1,arg_b2,arg_b3,arg_b4,arg_b5,arg_b7)
                filelist.append(pt_job)
        
            make_group(d_ws,filelist,Itag)
            '''

            #2）再提交2），提取结果
            
            for I in range(len(tab_res_cali)):
                iterid=tab_res_cali.Iite[I]
                simid=tab_res_cali.Isim[I]
                arg1=0.0153
                arg2=float(tab_res_cali.p2[I]) * scale_para_sen[0]
                arg3=float(tab_res_cali.p3[I]) * scale_para_sen[1]
                arg4=float(tab_res_cali.p4[I]) * scale_para_sen[2]
                arg5=float(tab_res_cali.p5[I]) * scale_para_sen[3]
                
                mod_extract(tab_res_cali.Igrid[I],d_outloc,Itag, iterid,simid, arg1,arg2,arg3,arg4,arg5)
            

