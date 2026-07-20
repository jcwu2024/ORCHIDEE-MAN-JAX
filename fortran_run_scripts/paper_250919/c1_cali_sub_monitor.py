#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
from datetime import datetime
import time
import pandas as pd
import subprocess
import logging

# In[ ]:


if __name__ == "__main__":
    
    pt_log='./log_main.log'
    logging.basicConfig(filename=pt_log,level=logging.INFO)

    #print('---------------------------Calibration start!---------------------------')
    logging.info('---------------------------Calibration start!---------------------------')
    
    d_ws='/public/share/qhcess/qhcess2/User/zzhao/script/orc_cali_240430/'
    tab_grid=pd.read_csv(d_ws+'tab_grid_all.csv')
    #tab_grid=pd.read_csv(d_ws+'tab_grid_rep.csv')
    #tab_grid=pd.read_csv(d_ws+'tab_grid_test.csv')
    
    #ls_grid=tab_grid.Igrid[0:10]
    #ls_grid=tab_grid.Igrid[10:20]
    #ls_grid=tab_grid.Igrid[20:40]
    #ls_grid=tab_grid.Igrid[40:100]
    #ls_grid=tab_grid.Igrid[100:170]
    #ls_grid=tab_grid.Igrid[170:240]
    #ls_grid=tab_grid.Igrid[240:310]
    #ls_grid=tab_grid.Igrid[310:380]
    #ls_grid=tab_grid.Igrid[380:410]
    #ls_grid=tab_grid.Igrid[410:480]
    #ls_grid=tab_grid.Igrid[480:500]
    #ls_grid=tab_grid.Igrid[500:520]
    #ls_grid=tab_grid.Igrid[520:540]
    #ls_grid=tab_grid.Igrid[540:560]
    #ls_grid=tab_grid.Igrid[560:580]
    #ls_grid=tab_grid.Igrid[580:650]
    #ls_grid=tab_grid.Igrid[650:len(tab_grid)]
    #ls_grid=tab_grid.Igrid[0:20]
    #ls_grid=tab_grid.Igrid[20:120]
    #ls_grid=tab_grid.Igrid[120:180]
    #ls_grid=tab_grid.Igrid[180:200]
    ls_grid=tab_grid.Igrid[200:240]
    #ls_grid=tab_grid.Igrid[100:len(tab_grid)]
    #ls_grid=['097.5-117.5'] 
    #ls_grid=['338.5-096.5'] 
    #ls_grid=['332.5-064.5'] 
    #ls_grid=['267.0-111.0','285.0-099.0','291.0-109.0','305.0-101.0'] 
    
    #(1)Job提交
    for Igrid in ls_grid:
    
        #Igrid='266.5-110.5'
        
        d_outf=d_ws+'outfiles/'
        #if not os.path.exists(d_outf):
        #    os.makedirs(d_outf)
        pt_outf=d_outf+'out_'+Igrid+'.txt'
        txt_add1='#SBATCH -o '+pt_outf

        txt_add2='export I_MPI_HYDRA_TOPOLIB=ipl'

        #txt_sub='bsub -q q_x86_share -n 1 -o '+pt_outf+' -J sub_'+Igrid+' python example_4p.py '+Igrid
        #!!!!!!修改提交方式，需要加上文件头
        #txt_add3='srun --exclusive  python ./c0_pods_4p.py '+Igrid
        txt_add3='srun python ./c0_pods_4p.py '+Igrid

        f_sub='./subfiles/sub_'+Igrid
        txt_sh1='cp -r sub_temp.sh '+f_sub
        os.system(txt_sh1)

        f = open(f_sub, 'a')
        f.write(txt_add1)
        f.write('\n'+txt_add2)
        f.write('\n'+txt_add3)
        f.close()

        #time.sleep(3)
        txt_sh2='sbatch '+f_sub
        #print(txt_sh2)
        os.system(txt_sh2)

    
    #print(datetime.now())
    #print('Job submit complete---------------------------')
    logging.info(datetime.now())
    logging.info('Job submit complete---------------------------')
    
    #(2)Job监控
    #while 1:
    ##循环到全部结束为止
    #    #print(datetime.now())
    #    #print('Job monitor start---------------------------')
    #    logging.info(datetime.now())
    #    logging.info('Job monitor start---------------------------')
    #    time.sleep(60)
    #    
    #    d_sig_break=d_ws+'sig_break/'
    #    if os.path.exists(d_sig_break):
    #      break
    #    
    #    ls_end_flag=[]
    #    
    #    for Igrid in ls_grid:
    #        
    #        #Igrid='266.5-110.5'

    #        #判断任务是否已在列表中
    #        sub_flag=0

    #        txt_bjobs='bjobs -l -J sub_'+Igrid
    #        bjobs_job=str(subprocess.check_output(txt_bjobs,shell=True))

    #        #print bjobs_job

    #        if bjobs_job=='No match record found!\n':
    #            stat_job='no job'
    #        else:
    #            stat_job=bjobs_job.split(', ')[2]

    #        if not (stat_job=='Status<RUN>') | (stat_job=='Status<PEND>') | (stat_job=='Status<STARTING>'):
    #            sub_flag=1 #没有任务


    #        #判断任务是否结束
    #        end_flag=0
    #        
    #        f = open(d_ws+'logfiles/log_'+Igrid+'.log', 'r')
    #        last_line=f.readlines()[-1]
    #        
    #        if not last_line=='-----Calibration complete-----':
    #            end_flag=1 #任务未完成


    #        #如果任务不在列表中，且没有结束，则重新提交
    #        if sub_flag & end_flag:
    #            
    #            pt_outf=d_outf+'out_'+Igrid+'.txt'
    #            txt_sub='bsub -q q_x86_share -n 1 -o '+pt_outf+' -J sub_'+Igrid+' python example.py '+Igrid
    #            os.system(txt_sub)

    #            #print 'resub: '+Igrid
    #            txt_log='-----!!!!!Resub: '+Igrid+'!!!!!------'
    #            logging.info(txt_log)
    #        
    #        ls_end_flag.append(end_flag)
    #        
    #        #print 'grid: '+Igrid+' sub:',sub_flag,' end:',end_flag
    #        txt_log='grid: '+Igrid+' sub:',sub_flag,' end:',end_flag
    #        logging.info(txt_log)

    #    #print(datetime.now())
    #    #print('Job monitor end---------------------------')
    #    logging.info(datetime.now())
    #    logging.info('Job monitor end---------------------------')
    #    
    #    if not any(ls_end_flag):
    #        break
            
    
    #print('---------------------------All clibration job complete!---------------------------')
    logging.info('---------------------------All clibration job complete!---------------------------')
    #return 1



