#!/afs/cern.ch/project/uslarp/opt/lxplus64/Python-2.7.2/bin/python

#----------------------------------------------------------------------
#                                                                      
#                           CERN                                       
#                                                                      
#     European Organization for Nuclear Research                       
#                                                                      
#     
#     This file is part of the code:
#                                                                      		            
#		           PyECLOUD Version 4.0                   
#                  
#                                                                       
#     Author and contact:   Giovanni IADAROLA 
#                           BE-ABP Group                               
#                           CERN                                       
#                           CH-1211 GENEVA 23                          
#                           SWITZERLAND  
#                           giovanni.iadarola@cern.ch                  
#                                                                      
#                contact:   Giovanni RUMOLO                            
#                           BE-ABP Group                               
#                           CERN                                      
#                           CH-1211 GENEVA 23                          
#                           SWITZERLAND  
#                           giovanni.rumolo@cern.ch                    
#                                                                      
#
#                                                                      
#     Copyright  CERN,  Geneva  2011  -  Copyright  and  any   other   
#     appropriate  legal  protection  of  this  computer program and   
#     associated documentation reserved  in  all  countries  of  the   
#     world.                                                           
#                                                                      
#     Organizations collaborating with CERN may receive this program   
#     and documentation freely and without charge.                     
#                                                                      
#     CERN undertakes no obligation  for  the  maintenance  of  this   
#     program,  nor responsibility for its correctness,  and accepts   
#     no liability whatsoever resulting from its use.                  
#                                                                      
#     Program  and documentation are provided solely for the use  of   
#     the organization to which they are distributed.                  
#                                                                      
#     This program  may  not  be  copied  or  otherwise  distributed   
#     without  permission. This message must be retained on this and   
#     any other authorized copies.                                     
#                                                                      
#     The material cannot be sold. CERN should be  given  credit  in   
#     all references.                                                  
#----------------------------------------------------------------------


import init as init

######
print 'PyECLOUD Version 4.0'

beamtim,MP_e, dynamics,impact_man, pyeclsaver, \
        gas_ion_flag, resgasion, t_ion, \
        spacech_ele,t_sc_ON, photoem_flag, phemiss,\
        flag_presence_sec_beams, sec_beams_list=init.read_input_files_and_init_components()
        
#add 1 OK step 2 OK field 3 OK ion 4 OK photoem 5 save profiles

print 'Start timestep iter'

## simulation
while not beamtim.end_simulation():
    
    beamtim.next_time_step()
    
    if flag_presence_sec_beams:
        for sec_beam in sec_beams_list:
            sec_beam.next_time_step()
    
    
    ## compute beam electric field (main and secondary beams)
    Ex_n_beam, Ey_n_beam = beamtim.get_beam_eletric_field(MP_e)
    
    if flag_presence_sec_beams:
        for sec_beam in sec_beams_list:
            Ex_n_secbeam, Ey_n_secbeam = sec_beam.get_beam_eletric_field(MP_e)
            Ex_n_beam+=Ex_n_secbeam
            Ey_n_beam+=Ey_n_secbeam
    
    ## compute electron space charge electric field
    Ex_sc_n, Ey_sc_n = spacech_ele.get_sc_eletric_field(MP_e)
    
    ## Total electric field
    Ex_n=Ex_sc_n+Ex_n_beam;
    Ey_n=Ey_sc_n+Ey_n_beam;
        
    ## save position before motion step
    old_pos=MP_e.get_positions()
    
    ## motion
    MP_e = dynamics.step(MP_e, Ex_n, Ey_n);
    
    ## impacts: backtracking and secondary emission
    MP_e = impact_man.backtrack_and_second_emiss(old_pos, MP_e)
 
    
    ## gas ionization (main and secondary beams)
    if(beamtim.tt_curr<t_ion and gas_ion_flag==1):
        MP_e = resgasion.generate(MP_e, beamtim.lam_t_curr, beamtim.Dt,beamtim.sigmax, beamtim.sigmay, 
                                x_beam_pos = beamtim.x_beam_pos, y_beam_pos = beamtim.y_beam_pos)
        if flag_presence_sec_beams:
            for sec_beam in sec_beams_list:
                MP_e = resgasion.generate(MP_e, sec_beam.lam_t_curr, sec_beam.Dt,sec_beam.sigmax, sec_beam.sigmay, 
                                x_beam_pos = sec_beam.x_beam_pos, y_beam_pos = sec_beam.y_beam_pos)
        
                
     
    ## photoemission (main and secondary beams)  
    if(photoem_flag==1):
        lam_curr_phem = beamtim.lam_t_curr
        if flag_presence_sec_beams:
            for sec_beam in sec_beams_list:
                lam_curr_phem += sec_beam.lam_t_curr
        MP_e = phemiss.generate(MP_e, lam_curr_phem, beamtim.Dt)

        
    ## Compute space charge field
    if (beamtim.tt_curr>t_sc_ON):
        spacech_ele.recompute_spchg_efield(MP_e, t_curr=beamtim.tt_curr)
        
    
    ## savings
    impact_man = pyeclsaver.witness(MP_e, beamtim, spacech_ele, impact_man, dynamics,gas_ion_flag,
                                    resgasion,t_ion,t_sc_ON, photoem_flag, phemiss,
                                    flag_presence_sec_beams,sec_beams_list)
    
    
    ## every bunch passage
    if beamtim.flag_new_bunch_pass:

        ## clean
        MP_e.clean_small_MPs()
        
        ## regeneration
        MP_e.check_for_regeneration()
        
        ## soft regeneration
        MP_e.check_for_soft_regeneration()        
        
        print '**** Done pass_numb = %d/%d\n'%(beamtim.pass_numb,beamtim.N_pass_tot)
    
