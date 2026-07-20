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




def extract_file(d_modelout_extract, Iage,arg1,arg2,arg3,arg4,arg5):

    #这里由于打包的时候把路径也打包了，所以解压路径比较长
    #os.system只能在当前路径下操作
    #txt_tem1 = d_outloc+Igrid+'/I'+str(iterid)
    #d_tar = txt_tem1+'/S'+str(simid)+'_'+str(arg2)[:6]+'_'+str(arg3)[:6]+'_'+str(arg4)[:6]+'_'+str(arg5)[:6]+'.tar'
    #d_outloc_grid = '/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_240430'+txt_tem1+'/S'+str(simid)+'_'+str(arg2)[:6]+'_'+str(arg3)[:6]+'_'+str(arg4)[:6]+'_'+str(arg5)[:6]+'/'

    #os.system('cd '+txt_tem1)
    #os.system('pwd')
    #os.system('tar xf '+d_tar)

    pt_nc=d_modelout_extract+'stomate_history_'+str(int(1961+Iage-1))+'.nc'

    #把最优结果拷贝到另一个文件夹
    #剩余的手动删除

    print 'out:d_modelout_extract'
    print(d_modelout_extract)
    print 'out:pt_nc'
    print(pt_nc)

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




def mod_extract(Igrid,d_ws,d_outloc,iterid,simid, arg1,arg2,arg3,arg4,arg5):
   
    d_modelout_best_grid=d_ws+'modelout_best/'+Igrid+'/'
    d_outloc_best=d_outloc+Igrid+'/I'+str(iterid)+'/'
    #nmf_modelout_best='S'+str(simid)+'_'+str(arg2)[:6]+'_'+str(arg3)[:6]+'_'+str(arg4)[:6]+'_'+str(arg5)[:6]

    if not os.path.exists(d_modelout_best_grid):
        os.makedirs(d_modelout_best_grid)

    print('--------------------------------------')
    print(d_outloc_best)
    os.chdir(d_outloc_best)

    #为避免文件名细微差别导致无法读取，修改文件名为best
    #txt1='cp '+nmf_modelout_best+'.tar '+d_modelout_best_grid+nmf_modelout_best+'.tar'
    txt1='cp S'+str(simid)+'_* '+d_modelout_best_grid+'best.tar'
    print(txt1)
    os.system(txt1)

    print(d_modelout_best_grid)
    os.chdir(d_modelout_best_grid)

    #txt2='tar xf '+nmf_modelout_best+'.tar'
    txt2='tar xf best.tar'
    print(txt2)
    os.system(txt2)

    txt3='mv S'+str(simid)+'_* best'
    os.system(txt3)

    #d_modelout_extract=d_modelout_best_grid+nmf_modelout_best+'/'
    d_modelout_extract=d_modelout_best_grid+'best/'


    p_std_AGB=[0, 1]
    p_std_BGB=[0, 1]
    p_std_GPP=[0, 1]
    p_std_NPP=[0, 1]

    #此处为没有归一化的观测数据列表
    dt_obs=pd.read_csv('/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_240430/dt_obs_gf_extglo.csv')    
    dt_obs_grid=dt_obs[dt_obs.Igrid==Igrid]
    dt_obs_grid=dt_obs_grid.reset_index(drop=True)
   
    dt_obs_grid['AGB_model']=np.nan
    dt_obs_grid['BGB_model']=np.nan
    dt_obs_grid['GPP_model']=np.nan
    dt_obs_grid['NPP_model']=np.nan
    
    for I in range(len(dt_obs_grid)):
        
        res_extract=extract_file(d_modelout_extract, dt_obs_grid.age[I],arg1,arg2,arg3,arg4,arg5)
        
        print(res_extract)

        dt_obs_grid.AGB_model[I]=res_extract[0]
        dt_obs_grid.BGB_model[I]=res_extract[1]
        dt_obs_grid.GPP_model[I]=res_extract[2]
        dt_obs_grid.NPP_model[I]=res_extract[3]
        
    dt_obs_grid.AGB_model=(dt_obs_grid.AGB_model*0.02 - p_std_AGB[0]) / p_std_AGB[1] #全部用归一化结果
    dt_obs_grid.BGB_model=(dt_obs_grid.BGB_model*0.02 - p_std_BGB[0]) / p_std_BGB[1]
    dt_obs_grid.GPP_model=(dt_obs_grid.GPP_model - p_std_GPP[0]) / p_std_GPP[1]
    dt_obs_grid.NPP_model=(dt_obs_grid.NPP_model - p_std_NPP[0]) / p_std_NPP[1]
    
    
    dt_obs_grid.to_csv('/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_240430/modelout_pods/modelout_'+Igrid+'.csv')



if __name__ == "__main__":

    #提交和提取
    #提交放入函数
    
    d_ws='/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_240430/'
    d_outloc='/public/share/qhcess/qhcess2/User/zzhao/OUT//orc_calibrate_240430/'
    #iterid=999
    #simid=999
    arg1=0.0153
    #arg2=50
    #arg3=0.2
    #arg4=0.2
    #arg5=30

    #1)先提交1），制作脚本

    #tab_grid=pd.read_csv(d_ws+'tab_grid.csv')  #用rep做出来的group也叫G0，所以先把之前的G0改名
    #ls_grid=tab_grid.Igrid

    #filelist=[]
    #for Igrid in ls_grid:
    #    pt_job=make_job(d_ws,d_outloc,Igrid,iterid,simid, arg1,arg2,arg3,arg4,arg5)
    #    filelist.append(pt_job)


    #make_group(d_ws,filelist)

    #2）再提交2），提取结果
    #for Igrid in ls_grid:
    #    mod_extract(Igrid,d_outloc,iterid,simid, arg1,arg2,arg3,arg4,arg5)

    tab_res_cali=pd.read_csv(d_ws+'tab_res_cali.csv',dtype='str')

    #for I in range(30):
    for I in range(len(tab_res_cali)):
    #for I in range(3,4):
        mod_extract(tab_res_cali.Igrid[I]  ,d_ws, d_outloc, tab_res_cali.Iite[I], tab_res_cali.Isim[I], arg1, tab_res_cali.p2[I], tab_res_cali.p3[I], tab_res_cali.p4[I], tab_res_cali.p5[I])


