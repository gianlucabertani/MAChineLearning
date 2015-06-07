//
//  MLReal.h
//  MAChineLearning
//
//  Created by Gianluca Bertani on 23/04/15.
//  Copyright (c) 2015 Gianluca Bertani. All rights reserved.
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

#ifndef MAChineLearning_MLReal_h
#define MAChineLearning_MLReal_h


/* Uncomment to use double precision.
 * Beware: it is much slower.
 
typedef double          MLReal;

#define ML_VDSP_VCLR    vDSP_vclrD
#define ML_VDSP_VTHRSC  vDSP_vthrscD
#define ML_VDSP_VTHRES  vDSP_vthresD
#define ML_VDSP_VSMUL   vDSP_vsmulD
#define ML_VDSP_VSDIV   vDSP_vsdivD
#define ML_VDSP_SVDIV   vDSP_svdivD
#define ML_VDSP_VSADD   vDSP_vsaddD
#define ML_VDSP_DOTPR   vDSP_dotprD
#define ML_VDSP_VSQ     vDSP_vsqD
#define ML_VDSP_VMUL    vDSP_vmulD
#define ML_VDSP_VADD    vDSP_vaddD
#define ML_VDSP_VSUB    vDSP_vsubD
#define ML_VDSP_VDIV    vDSP_vdivD
#define ML_VDSP_VSMA    vDSP_vsmaD
#define ML_VDSP_SVE		vDSP_sveD
#define ML_VDSP_SVESQ   vDSP_svesqD
#define ML_VDSP_VGATHRA vDSP_vgathraD
#define ML_VDSP_SVE     vDSP_sveD
#define ML_VDSP_DIST    vDSP_vdistD

#define ML_VVEXP        vvexp
 
#define ML_SQRT         sqrt
 
 */


typedef float           MLReal;

#define ML_VDSP_VCLR    vDSP_vclr
#define ML_VDSP_VTHRSC  vDSP_vthrsc
#define ML_VDSP_VTHRES  vDSP_vthres
#define ML_VDSP_VSMUL   vDSP_vsmul
#define ML_VDSP_VSDIV   vDSP_vsdiv
#define ML_VDSP_SVDIV   vDSP_svdiv
#define ML_VDSP_VSADD   vDSP_vsadd
#define ML_VDSP_DOTPR   vDSP_dotpr
#define ML_VDSP_VSQ     vDSP_vsq
#define ML_VDSP_VMUL    vDSP_vmul
#define ML_VDSP_VADD    vDSP_vadd
#define ML_VDSP_VSUB    vDSP_vsub
#define ML_VDSP_VDIV    vDSP_vdiv
#define ML_VDSP_VSMA    vDSP_vsma
#define ML_VDSP_SVE		vDSP_sve
#define ML_VDSP_SVESQ   vDSP_svesq
#define ML_VDSP_VGATHRA vDSP_vgathra
#define ML_VDSP_SVE     vDSP_sve
#define ML_VDSP_DIST    vDSP_vdist

#define ML_VVEXP        vvexpf

#define ML_SQRT         sqrtf


#endif
