//
//  NeuralNetworkReal.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 03/04/15.
//  Copyright (c) 2015 Flying Dolphin Studio. All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions
//  are met:
//
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of Gianluca Bertani nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//

#ifndef MAChineLearning_NeuralNetworkReal_h
#define MAChineLearning_NeuralNetworkReal_h


/* Uncomment to use double precision
 
typedef double        nnREAL;

#define nnVDSP_VCLR   vDSP_vclrD
#define nnVDSP_VTHRSC vDSP_vthrscD
#define nnVDSP_VTHRES vDSP_vthresD
#define nnVDSP_VSMUL  vDSP_vsmulD
#define nnVDSP_VSDIV  vDSP_vsdivD
#define nnVDSP_SVDIV  vDSP_svdivD
#define nnVDSP_VSADD  vDSP_vsaddD
#define nnVDSP_DOTPR  vDSP_dotprD
#define nnVDSP_VSQ    vDSP_vsqD
#define nnVDSP_VMUL   vDSP_vmulD
#define nnVDSP_VADD   vDSP_vaddD
#define nnVDSP_VSUB   vDSP_vsubD
#define nnVDSP_VDIV   vDSP_vdivD
#define nnVDSP_VSMA   vDSP_vsmaD
#define nnVDSP_SVESQ  vDSP_svesqD

#define nnVVEXP       vvexp
#define nnVVREC       vvrec
#define nnVVTANH      vvtanh

 */


typedef float         nnREAL;

#define nnVDSP_VCLR   vDSP_vclr
#define nnVDSP_VTHRSC vDSP_vthrsc
#define nnVDSP_VTHRES vDSP_vthres
#define nnVDSP_VSMUL  vDSP_vsmul
#define nnVDSP_VSDIV  vDSP_vsdiv
#define nnVDSP_SVDIV  vDSP_svdiv
#define nnVDSP_VSADD  vDSP_vsadd
#define nnVDSP_DOTPR  vDSP_dotpr
#define nnVDSP_VSQ    vDSP_vsq
#define nnVDSP_VMUL   vDSP_vmul
#define nnVDSP_VADD   vDSP_vadd
#define nnVDSP_VSUB   vDSP_vsub
#define nnVDSP_VDIV   vDSP_vdiv
#define nnVDSP_VSMA   vDSP_vsma
#define nnVDSP_SVESQ  vDSP_svesq

#define nnVVEXP       vvexpf
#define nnVVREC       vvrecf
#define nnVVTANH      vvtanhf


#endif
