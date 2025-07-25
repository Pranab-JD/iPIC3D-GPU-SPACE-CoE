/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #ifndef GMRES_new2_H
 #define GMRES_new2_H
 
 #include "ipicfwd.h"
 
 typedef void (EMfields3D::*FIELD_IMAGE) (double *, double *);
 typedef void (*GENERIC_IMAGE) (double *, double *);
 
 void GMRES(FIELD_IMAGE FunctionImage, double *xkrylov, int xkrylovlen, const double *b, int m, int max_iter, double tol, EMfields3D * field);
 
 void GMRESasPreconditioner(FIELD_IMAGE FunctionImage, double *xkrylov, int xkrylovlen, const double *b, int m, int max_iter, double tol, EMfields3D * field,
                            double *rPrec, double *imPrec, double *sPrec, double *csPrec, double *snPrec, double *yPrec, double **HPrec, double **VPrec);
 void GMRESasPreconditionerNoComm(FIELD_IMAGE FunctionImage, double *xkrylov, int xkrylovlen, const double *b, int m, int max_iter, double tol, EMfields3D * field);
 void FGMRES(FIELD_IMAGE FunctionImage, FIELD_IMAGE LocalFunctionImage, double *xkrylov, int xkrylovlen, const double *b, int m, int max_iter, double tol, EMfields3D * field);
 
 void ApplyPlaneRotation(double &dx, double &dy, double &cs, double &sn);
 
 #endif
 