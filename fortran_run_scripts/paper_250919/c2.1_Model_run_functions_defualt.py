#!/usr/bin/env python
# coding: utf-8

# In[36]:


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


# In[ ]:


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




def mod_extract(Igrid,d_outloc,iterid,simid, arg1,arg2,arg3,arg4,arg5):
    
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
    
    
    dt_obs_grid.to_csv('/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_250919/modelout_default/modelout_'+Igrid+'.csv')


def make_job(d_ws,d_outloc,Igrid,iterid,simid, arg1,arg2,arg3,arg4,arg5):
    

    d_grid_job = d_ws+'Point/'+Igrid+'/I'+str(iterid)+'/S'+str(simid)+'_'+str(arg2)[:6]+'_'+str(arg3)[:6]+'_'+str(arg4)[:6]+'_'+str(arg5)[:6]+'/'
    if not os.path.exists(d_grid_job):
        os.makedirs(d_grid_job)

        
    #(1)复制并修改run.def
    rundname = 'run.def_'+str(arg2)[:6]+'_'+str(arg3)[:6]+'_'+str(arg4)[:6]+'_'+str(arg5)[:6]
    pt_rund = d_grid_job + rundname
    txt_cp1 = 'cp -r run.def.vn '+pt_rund
    print txt_cp1
    os.system(txt_cp1)
   
 
    func_replace(pt_rund, 'ARJV__00014'   , 12 , 2.59)
    func_replace(pt_rund, 'SLA__00014'    , 11 , arg1)
    func_replace(pt_rund, 'SLA_MAX__00014', 15 , arg1)
    func_replace(pt_rund, 'SLA_MIN__00014', 15 , arg1)

    func_replace(pt_rund, 'VCMAX25__00014', 15 , arg2)
    func_replace(pt_rund, 'MAINT_RESP_SLOPE_C__00014', 26 , arg3)
    func_replace(pt_rund, 'ALLOC_MIN__00014',          17 , arg4)
    func_replace(pt_rund, 'RESIDENCE_TIME__00014'    , 22 , arg5)



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
    
    

def make_group(d_ws,filelist):
    nproc = 32
    nproc2= 32

    filelist = np.sort(filelist)

    nsite = len(filelist)
    #----divide into groups based on nproc----------
    ngroup = int((nsite+nproc2-1)/nproc2)
    d_grpjob = d_ws + '/Group_job_default/'+str(arg1)[:5]+'_'+str(arg2)[:5]+'_'+str(arg3)[:5]+'/'
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
                line_tmp  = job + ' &\n'
                f.writelines(line_tmp)
            f.writelines('wait')
        f.close()

    #txt_chmod='chmod 777 '+d_alljob+'*'
    #os.system(txt_chmod)
    txt_chmod='chmod 777 '+d_grpjob+'*'
    os.system(txt_chmod)

    for i in np.arange(ngroup):
        Gjobname = 'G' + str(i)
        groupfile = d_grpjob + Gjobname + '.sh'
        #txt_sub='qsub '+groupfile
        #txt_sub='bsub -q q_x86_share_1 -n '+str(nproc)+' '+groupfile
        txt_sub='bsub -q q_x86_share -n 1 '+' '+groupfile





if __name__ == "__main__":

    #提交和提取
    #提交放入函数
    
    d_ws='/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_250919/'
    d_outloc='/public/share/qhcess/qhcess2/User/zzhao/OUT/orc_calibrate_250919_default/'
    iterid=999
    simid=999
    arg1=0.0153
    arg2=50
    arg3=0.2
    arg4=0.2
    arg5=30

    #1)先提交1），制作脚本

    #tab_grid=pd.read_csv(d_ws+'tab_grid.csv')  #用rep做出来的group也叫G0，所以先把之前的G0改名
    tab_grid=pd.read_csv(d_ws+'tab_grid_all.csv')  #用rep做出来的group也叫G0，所以先把之前的G0改名
    ls_grid=tab_grid.Igrid

    filelist=[]
    for Igrid in ls_grid:
        pt_job=make_job(d_ws,d_outloc,Igrid,iterid,simid, arg1,arg2,arg3,arg4,arg5)
        filelist.append(pt_job)


    make_group(d_ws,filelist)

    #2）再提交2），提取结果
    #for Igrid in ls_grid:
    #    mod_extract(Igrid,d_outloc,iterid,simid, arg1,arg2,arg3,arg4,arg5)



