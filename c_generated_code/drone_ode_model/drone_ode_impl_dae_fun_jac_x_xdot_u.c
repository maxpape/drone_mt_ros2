/* This file was automatically generated by CasADi 3.6.5.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) drone_ode_impl_dae_fun_jac_x_xdot_u_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

static const casadi_int casadi_s0[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[31] = {27, 1, 0, 27, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
static const casadi_int casadi_s4[59] = {13, 13, 0, 0, 0, 0, 5, 11, 17, 22, 23, 24, 25, 31, 37, 43, 4, 5, 6, 7, 8, 3, 5, 6, 7, 8, 9, 3, 4, 6, 7, 8, 9, 3, 4, 5, 7, 8, 0, 1, 2, 3, 4, 5, 6, 11, 12, 3, 4, 5, 6, 10, 12, 3, 4, 5, 6, 10, 11};
static const casadi_int casadi_s5[29] = {13, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s6[31] = {13, 4, 0, 6, 12, 18, 24, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12};

/* drone_ode_impl_dae_fun_jac_x_xdot_u:(i0[13],i1[13],i2[4],i3[],i4[],i5[27])->(o0[13],o1[13x13,43nz],o2[13x13,13nz],o3[13x4,24nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a5, a6, a7, a8, a9;
  a0=arg[1]? arg[1][0] : 0;
  a1=arg[0]? arg[0][7] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[1]? arg[1][1] : 0;
  a1=arg[0]? arg[0][8] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[1]? arg[1][2] : 0;
  a1=arg[0]? arg[0][9] : 0;
  a0=(a0-a1);
  if (res[0]!=0) res[0][2]=a0;
  a0=arg[1]? arg[1][3] : 0;
  a1=5.0000000000000000e-01;
  a2=arg[0]? arg[0][4] : 0;
  a3=arg[0]? arg[0][10] : 0;
  a4=(a2*a3);
  a5=arg[0]? arg[0][5] : 0;
  a6=arg[0]? arg[0][11] : 0;
  a7=(a5*a6);
  a4=(a4+a7);
  a7=arg[0]? arg[0][6] : 0;
  a8=arg[0]? arg[0][12] : 0;
  a9=(a7*a8);
  a4=(a4+a9);
  a4=(a1*a4);
  a0=(a0+a4);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1]? arg[1][4] : 0;
  a4=arg[0]? arg[0][3] : 0;
  a9=(a4*a3);
  a10=(a7*a6);
  a9=(a9-a10);
  a10=(a5*a8);
  a9=(a9+a10);
  a9=(a1*a9);
  a0=(a0-a9);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[1]? arg[1][5] : 0;
  a9=(a7*a3);
  a10=(a4*a6);
  a9=(a9+a10);
  a10=(a2*a8);
  a9=(a9-a10);
  a9=(a1*a9);
  a0=(a0-a9);
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[1]? arg[1][6] : 0;
  a9=(a2*a6);
  a10=(a5*a3);
  a9=(a9-a10);
  a10=(a4*a8);
  a9=(a9+a10);
  a9=(a1*a9);
  a0=(a0-a9);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a9=2.;
  a10=(a9*a2);
  a11=(a10*a7);
  a12=(a9*a5);
  a13=(a12*a4);
  a11=(a11+a13);
  a13=arg[2]? arg[2][0] : 0;
  a14=arg[2]? arg[2][1] : 0;
  a15=(a13+a14);
  a16=arg[2]? arg[2][2] : 0;
  a15=(a15+a16);
  a17=arg[2]? arg[2][3] : 0;
  a15=(a15+a17);
  a18=(a11*a15);
  a19=arg[5]? arg[5][0] : 0;
  a18=(a18/a19);
  a0=(a0-a18);
  if (res[0]!=0) res[0][7]=a0;
  a0=arg[1]? arg[1][8] : 0;
  a18=(a9*a5);
  a20=(a18*a7);
  a21=(a9*a2);
  a22=(a21*a4);
  a20=(a20-a22);
  a22=(a20*a15);
  a22=(a22/a19);
  a0=(a0-a22);
  if (res[0]!=0) res[0][8]=a0;
  a0=arg[1]? arg[1][9] : 0;
  a22=1.;
  a23=casadi_sq(a2);
  a23=(a9*a23);
  a23=(a22-a23);
  a24=casadi_sq(a5);
  a24=(a9*a24);
  a23=(a23-a24);
  a24=(a23*a15);
  a24=(a24/a19);
  a25=arg[5]? arg[5][1] : 0;
  a24=(a24+a25);
  a0=(a0-a24);
  if (res[0]!=0) res[0][9]=a0;
  a0=arg[1]? arg[1][10] : 0;
  a24=arg[5]? arg[5][7] : 0;
  a25=(a24*a16);
  a26=arg[5]? arg[5][5] : 0;
  a27=(a26*a13);
  a28=arg[5]? arg[5][6] : 0;
  a29=(a28*a14);
  a27=(a27+a29);
  a25=(a25-a27);
  a27=arg[5]? arg[5][8] : 0;
  a29=(a27*a17);
  a25=(a25+a29);
  a29=arg[5]? arg[5][4] : 0;
  a30=(a29*a8);
  a31=(a6*a30);
  a32=arg[5]? arg[5][3] : 0;
  a33=(a32*a6);
  a34=(a8*a33);
  a31=(a31-a34);
  a25=(a25-a31);
  a31=arg[5]? arg[5][2] : 0;
  a25=(a25/a31);
  a0=(a0-a25);
  if (res[0]!=0) res[0][10]=a0;
  a0=arg[1]? arg[1][11] : 0;
  a25=arg[5]? arg[5][9] : 0;
  a34=(a25*a13);
  a35=arg[5]? arg[5][10] : 0;
  a36=(a35*a14);
  a34=(a34-a36);
  a36=(a35*a16);
  a34=(a34-a36);
  a36=arg[5]? arg[5][12] : 0;
  a37=(a36*a17);
  a34=(a34+a37);
  a37=(a31*a3);
  a38=(a8*a37);
  a39=(a3*a30);
  a38=(a38-a39);
  a34=(a34-a38);
  a34=(a34/a32);
  a0=(a0-a34);
  if (res[0]!=0) res[0][11]=a0;
  a0=arg[1]? arg[1][12] : 0;
  a34=arg[5]? arg[5][13] : 0;
  a14=(a34*a14);
  a13=(a34*a13);
  a14=(a14-a13);
  a16=(a34*a16);
  a14=(a14-a16);
  a17=(a34*a17);
  a14=(a14+a17);
  a17=(a3*a33);
  a16=(a6*a37);
  a17=(a17-a16);
  a14=(a14-a17);
  a14=(a14/a29);
  a0=(a0-a14);
  if (res[0]!=0) res[0][12]=a0;
  a0=(a1*a3);
  a0=(-a0);
  if (res[1]!=0) res[1][0]=a0;
  a0=(a1*a6);
  a0=(-a0);
  if (res[1]!=0) res[1][1]=a0;
  a0=(a1*a8);
  a0=(-a0);
  if (res[1]!=0) res[1][2]=a0;
  a12=(a15*a12);
  a12=(a12/a19);
  a12=(-a12);
  if (res[1]!=0) res[1][3]=a12;
  a21=(a15*a21);
  a21=(a21/a19);
  if (res[1]!=0) res[1][4]=a21;
  a21=(a1*a3);
  if (res[1]!=0) res[1][5]=a21;
  a21=(a1*a8);
  if (res[1]!=0) res[1][6]=a21;
  a21=(a1*a6);
  a21=(-a21);
  if (res[1]!=0) res[1][7]=a21;
  a21=(a9*a7);
  a21=(a15*a21);
  a21=(a21/a19);
  a21=(-a21);
  if (res[1]!=0) res[1][8]=a21;
  a21=(a9*a4);
  a21=(a15*a21);
  a21=(a21/a19);
  if (res[1]!=0) res[1][9]=a21;
  a21=(a2+a2);
  a21=(a9*a21);
  a21=(a15*a21);
  a21=(a21/a19);
  if (res[1]!=0) res[1][10]=a21;
  a21=(a1*a6);
  if (res[1]!=0) res[1][11]=a21;
  a21=(a1*a8);
  a21=(-a21);
  if (res[1]!=0) res[1][12]=a21;
  a21=(a1*a3);
  if (res[1]!=0) res[1][13]=a21;
  a21=(a9*a4);
  a21=(a15*a21);
  a21=(a21/a19);
  a21=(-a21);
  if (res[1]!=0) res[1][14]=a21;
  a21=(a9*a7);
  a21=(a15*a21);
  a21=(a21/a19);
  a21=(-a21);
  if (res[1]!=0) res[1][15]=a21;
  a21=(a5+a5);
  a9=(a9*a21);
  a9=(a15*a9);
  a9=(a9/a19);
  if (res[1]!=0) res[1][16]=a9;
  a9=(a1*a8);
  if (res[1]!=0) res[1][17]=a9;
  a9=(a1*a6);
  if (res[1]!=0) res[1][18]=a9;
  a9=(a1*a3);
  a9=(-a9);
  if (res[1]!=0) res[1][19]=a9;
  a10=(a15*a10);
  a10=(a10/a19);
  a10=(-a10);
  if (res[1]!=0) res[1][20]=a10;
  a15=(a15*a18);
  a15=(a15/a19);
  a15=(-a15);
  if (res[1]!=0) res[1][21]=a15;
  a15=-1.;
  if (res[1]!=0) res[1][22]=a15;
  if (res[1]!=0) res[1][23]=a15;
  if (res[1]!=0) res[1][24]=a15;
  a15=(a1*a2);
  if (res[1]!=0) res[1][25]=a15;
  a15=(a1*a4);
  a15=(-a15);
  if (res[1]!=0) res[1][26]=a15;
  a15=(a1*a7);
  a15=(-a15);
  if (res[1]!=0) res[1][27]=a15;
  a15=(a1*a5);
  if (res[1]!=0) res[1][28]=a15;
  a15=(a8*a31);
  a15=(a15-a30);
  a15=(a15/a32);
  if (res[1]!=0) res[1][29]=a15;
  a15=(a6*a31);
  a15=(a33-a15);
  a15=(a15/a29);
  if (res[1]!=0) res[1][30]=a15;
  a15=(a1*a5);
  if (res[1]!=0) res[1][31]=a15;
  a15=(a1*a7);
  if (res[1]!=0) res[1][32]=a15;
  a15=(a1*a4);
  a15=(-a15);
  if (res[1]!=0) res[1][33]=a15;
  a15=(a1*a2);
  a15=(-a15);
  if (res[1]!=0) res[1][34]=a15;
  a8=(a8*a32);
  a30=(a30-a8);
  a30=(a30/a31);
  if (res[1]!=0) res[1][35]=a30;
  a30=(a3*a32);
  a30=(a30-a37);
  a30=(a30/a29);
  if (res[1]!=0) res[1][36]=a30;
  a7=(a1*a7);
  if (res[1]!=0) res[1][37]=a7;
  a5=(a1*a5);
  a5=(-a5);
  if (res[1]!=0) res[1][38]=a5;
  a2=(a1*a2);
  if (res[1]!=0) res[1][39]=a2;
  a1=(a1*a4);
  a1=(-a1);
  if (res[1]!=0) res[1][40]=a1;
  a6=(a6*a29);
  a6=(a6-a33);
  a6=(a6/a31);
  if (res[1]!=0) res[1][41]=a6;
  a3=(a3*a29);
  a37=(a37-a3);
  a37=(a37/a32);
  if (res[1]!=0) res[1][42]=a37;
  if (res[2]!=0) res[2][0]=a22;
  if (res[2]!=0) res[2][1]=a22;
  if (res[2]!=0) res[2][2]=a22;
  if (res[2]!=0) res[2][3]=a22;
  if (res[2]!=0) res[2][4]=a22;
  if (res[2]!=0) res[2][5]=a22;
  if (res[2]!=0) res[2][6]=a22;
  if (res[2]!=0) res[2][7]=a22;
  if (res[2]!=0) res[2][8]=a22;
  if (res[2]!=0) res[2][9]=a22;
  if (res[2]!=0) res[2][10]=a22;
  if (res[2]!=0) res[2][11]=a22;
  if (res[2]!=0) res[2][12]=a22;
  a22=(a11/a19);
  a22=(-a22);
  if (res[3]!=0) res[3][0]=a22;
  a22=(a20/a19);
  a22=(-a22);
  if (res[3]!=0) res[3][1]=a22;
  a22=(a23/a19);
  a22=(-a22);
  if (res[3]!=0) res[3][2]=a22;
  a26=(a26/a31);
  if (res[3]!=0) res[3][3]=a26;
  a25=(a25/a32);
  a25=(-a25);
  if (res[3]!=0) res[3][4]=a25;
  a25=(a34/a29);
  if (res[3]!=0) res[3][5]=a25;
  a25=(a11/a19);
  a25=(-a25);
  if (res[3]!=0) res[3][6]=a25;
  a25=(a20/a19);
  a25=(-a25);
  if (res[3]!=0) res[3][7]=a25;
  a25=(a23/a19);
  a25=(-a25);
  if (res[3]!=0) res[3][8]=a25;
  a28=(a28/a31);
  if (res[3]!=0) res[3][9]=a28;
  a28=(a35/a32);
  if (res[3]!=0) res[3][10]=a28;
  a28=(a34/a29);
  a28=(-a28);
  if (res[3]!=0) res[3][11]=a28;
  a28=(a11/a19);
  a28=(-a28);
  if (res[3]!=0) res[3][12]=a28;
  a28=(a20/a19);
  a28=(-a28);
  if (res[3]!=0) res[3][13]=a28;
  a28=(a23/a19);
  a28=(-a28);
  if (res[3]!=0) res[3][14]=a28;
  a24=(a24/a31);
  a24=(-a24);
  if (res[3]!=0) res[3][15]=a24;
  a35=(a35/a32);
  if (res[3]!=0) res[3][16]=a35;
  a35=(a34/a29);
  if (res[3]!=0) res[3][17]=a35;
  a11=(a11/a19);
  a11=(-a11);
  if (res[3]!=0) res[3][18]=a11;
  a20=(a20/a19);
  a20=(-a20);
  if (res[3]!=0) res[3][19]=a20;
  a23=(a23/a19);
  a23=(-a23);
  if (res[3]!=0) res[3][20]=a23;
  a27=(a27/a31);
  a27=(-a27);
  if (res[3]!=0) res[3][21]=a27;
  a36=(a36/a32);
  a36=(-a36);
  if (res[3]!=0) res[3][22]=a36;
  a34=(a34/a29);
  a34=(-a34);
  if (res[3]!=0) res[3][23]=a34;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_u(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_u_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_u_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_jac_x_xdot_u_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_u_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_jac_x_xdot_u_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_jac_x_xdot_u_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_jac_x_xdot_u_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_impl_dae_fun_jac_x_xdot_u_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_impl_dae_fun_jac_x_xdot_u_n_out(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_impl_dae_fun_jac_x_xdot_u_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_impl_dae_fun_jac_x_xdot_u_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    case 5: return "i5";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_impl_dae_fun_jac_x_xdot_u_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_impl_dae_fun_jac_x_xdot_u_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s0;
    case 2: return casadi_s1;
    case 3: return casadi_s2;
    case 4: return casadi_s2;
    case 5: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_impl_dae_fun_jac_x_xdot_u_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    case 3: return casadi_s6;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_u_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 4;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_jac_x_xdot_u_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 4*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
