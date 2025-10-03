c      program prove_erf
c      real*8 x,y,wx,wy
c      
c      x=1.d0
c      y=2.d0
      
c      call ERRF(x, y, WX, WY)
c      write(*,*) 'x=',x,'y=',y,'wx=',wx,'wy=',wy
      
      
c      end program
      
      
      
      SUBROUTINE ERRF(XX, YY, WX, WY)
Cf2py intent(in)  XX   
Cf2py intent(in)  YY                                      
Cf2py intent(out) WX   
Cf2py intent(out) WY   
*----------------------------------------------------------------------*   
* Purpose:                                                             *   
*   Modification of WWERF, double precision complex error function,    *   
*   written at CERN by K. Koelbig.                                     *   
* Input:                                                               *   
*   XX, YY    (real)    Argument to CERF.                              *   
* Output:                                                              *   
*   WX, WY    (real)    Function result.                               *   
*----------------------------------------------------------------------*   
                                                                           
*---- Double precision version.                                            
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER (I-N)                   
      PARAMETER         (MWFLT = 2, MREAL = 4)                             
      PARAMETER         (MCWRD = 4)                                        
      PARAMETER         (MCNAM = 16, MWNAM = MCNAM / MCWRD)                
      PARAMETER         (MCFIL = 80, MCRNG = 40, MCSTR = 80)               
                                                                           
      PARAMETER         (CC     = 1.12837 91670 9551D0)                    
      PARAMETER         (ONE    = 1.D0)                                    
      PARAMETER         (TWO    = 2.D0)                                    
      PARAMETER         (XLIM   = 5.33D0)                                  
      PARAMETER         (YLIM   = 4.29D0)                                  
      DIMENSION         RX(33), RY(33)                                     
                                                                           
      X = ABS(XX)                                                          
      Y = ABS(YY)                                                          
                                                                           
      IF (Y .LT. YLIM  .AND.  X .LT. XLIM) THEN                            
        Q  = (ONE - Y / YLIM) * SQRT(ONE - (X/XLIM)**2)                    
        H  = ONE / (3.2D0 * Q)                                             
        NC = 7 + INT(23.0*Q)                                               
        XL = H**(1 - NC)                                                   
        XH = Y + 0.5D0/H                                                   
        YH = X                                                             
        NU = 10 + INT(21.0*Q)                                              
        RX(NU+1) = 0.                                                      
        RY(NU+1) = 0.                                                      
                                                                           
        DO 10 N = NU, 1, -1                                                
          TX = XH + N * RX(N+1)                                            
          TY = YH - N * RY(N+1)                                            
          TN = TX*TX + TY*TY                                               
          RX(N) = 0.5D0 * TX / TN                                          
          RY(N) = 0.5D0 * TY / TN                                          
   10   CONTINUE                                                           
                                                                           
        SX = 0.                                                            
        SY = 0.                                                            
                                                                           
        DO 20 N = NC, 1, -1                                                
          SAUX = SX + XL                                                   
          SX = RX(N) * SAUX - RY(N) * SY                                   
          SY = RX(N) * SY + RY(N) * SAUX                                   
          XL = H * XL                                                      
   20   CONTINUE                                                           
                                                                           
        WX = CC * SX                                                       
        WY = CC * SY                                                       
      ELSE                                                                 
        XH = Y                                                             
        YH = X                                                             
        RX(1) = 0.                                                         
        RY(1) = 0.                                                         
                                                                           
        DO 30 N = 9, 1, -1                                                 
          TX = XH + N * RX(1)                                              
          TY = YH - N * RY(1)                                              
          TN = TX*TX + TY*TY                                               
          RX(1) = 0.5D0 * TX / TN                                          
          RY(1) = 0.5D0 * TY / TN                                          
   30   CONTINUE                                                           
                                                                           
        WX = CC * RX(1)                                                    
        WY = CC * RY(1)                                                    
      ENDIF                                                                
                                                                           
      IF(Y .EQ. 0.) WX = EXP(-X**2)                                        
      IF(YY .LT. 0.) THEN                                                  
        WX =   TWO * EXP(Y*Y-X*X) * COS(TWO*X*Y) - WX                      
        WY = - TWO * EXP(Y*Y-X*X) * SIN(TWO*X*Y) - WY                      
        IF(XX .GT. 0.) WY = -WY                                            
      ELSE                                                                 
        IF(XX .LT. 0.) WY = -WY                                            
      ENDIF                                                                
                                                                           
      END
