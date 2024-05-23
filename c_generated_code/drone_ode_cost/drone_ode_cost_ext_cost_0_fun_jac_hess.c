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
  #define CASADI_PREFIX(ID) drone_ode_cost_ext_cost_0_fun_jac_hess_ ## ID
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
#define casadi_s7 CASADI_PREFIX(s7)
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

static const casadi_int casadi_s0[21] = {17, 1, 0, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[31] = {27, 1, 0, 27, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s5[25] = {21, 1, 0, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
static const casadi_int casadi_s6[47] = {21, 21, 0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10};
static const casadi_int casadi_s7[24] = {0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* drone_ode_cost_ext_cost_0_fun_jac_hess:(i0[17],i1[4],i2[],i3[27])->(o0,o1[21],o2[21x21,23nz],o3[],o4[0x21]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a4, a5, a6, a7, a8, a9;
  a0=150.;
  a1=arg[0]? arg[0][0] : 0;
  a2=arg[3]? arg[3][14] : 0;
  a3=(a1-a2);
  a3=(a0*a3);
  a1=(a1-a2);
  a2=(a3*a1);
  a4=arg[0]? arg[0][1] : 0;
  a5=arg[3]? arg[3][15] : 0;
  a6=(a4-a5);
  a6=(a0*a6);
  a4=(a4-a5);
  a5=(a6*a4);
  a2=(a2+a5);
  a5=2700.;
  a7=arg[0]? arg[0][2] : 0;
  a8=arg[3]? arg[3][16] : 0;
  a9=(a7-a8);
  a9=(a5*a9);
  a7=(a7-a8);
  a8=(a9*a7);
  a2=(a2+a8);
  a8=100.;
  a10=arg[3]? arg[3][17] : 0;
  a11=casadi_sq(a10);
  a12=arg[3]? arg[3][18] : 0;
  a13=casadi_sq(a12);
  a11=(a11+a13);
  a13=arg[3]? arg[3][19] : 0;
  a14=casadi_sq(a13);
  a11=(a11+a14);
  a14=arg[3]? arg[3][20] : 0;
  a15=casadi_sq(a14);
  a11=(a11+a15);
  a15=(a10/a11);
  a16=arg[0]? arg[0][6] : 0;
  a17=(a15*a16);
  a18=arg[0]? arg[0][3] : 0;
  a19=(a14/a11);
  a20=(a18*a19);
  a17=(a17-a20);
  a20=arg[0]? arg[0][5] : 0;
  a21=(a12/a11);
  a22=(a20*a21);
  a23=arg[0]? arg[0][4] : 0;
  a11=(a13/a11);
  a24=(a23*a11);
  a22=(a22-a24);
  a17=(a17+a22);
  a17=(a8*a17);
  a22=casadi_sq(a10);
  a24=casadi_sq(a12);
  a22=(a22+a24);
  a24=casadi_sq(a13);
  a22=(a22+a24);
  a24=casadi_sq(a14);
  a22=(a22+a24);
  a10=(a10/a22);
  a16=(a10*a16);
  a14=(a14/a22);
  a18=(a18*a14);
  a16=(a16-a18);
  a12=(a12/a22);
  a20=(a20*a12);
  a13=(a13/a22);
  a23=(a23*a13);
  a20=(a20-a23);
  a16=(a16+a20);
  a20=(a17*a16);
  a2=(a2+a20);
  a20=4.;
  a23=arg[1]? arg[1][0] : 0;
  a22=(a20*a23);
  a18=(a22*a23);
  a24=arg[1]? arg[1][1] : 0;
  a25=(a20*a24);
  a26=(a25*a24);
  a18=(a18+a26);
  a26=arg[1]? arg[1][2] : 0;
  a27=(a20*a26);
  a28=(a27*a26);
  a18=(a18+a28);
  a28=arg[1]? arg[1][3] : 0;
  a29=(a20*a28);
  a30=(a29*a28);
  a18=(a18+a30);
  a2=(a2+a18);
  if (res[0]!=0) res[0][0]=a2;
  a23=(a20*a23);
  a22=(a22+a23);
  if (res[1]!=0) res[1][0]=a22;
  a24=(a20*a24);
  a25=(a25+a24);
  if (res[1]!=0) res[1][1]=a25;
  a26=(a20*a26);
  a27=(a27+a26);
  if (res[1]!=0) res[1][2]=a27;
  a20=(a20*a28);
  a29=(a29+a20);
  if (res[1]!=0) res[1][3]=a29;
  a1=(a0*a1);
  a3=(a3+a1);
  if (res[1]!=0) res[1][4]=a3;
  a0=(a0*a4);
  a6=(a6+a0);
  if (res[1]!=0) res[1][5]=a6;
  a5=(a5*a7);
  a9=(a9+a5);
  if (res[1]!=0) res[1][6]=a9;
  a9=(a14*a17);
  a16=(a8*a16);
  a5=(a19*a16);
  a9=(a9+a5);
  a9=(-a9);
  if (res[1]!=0) res[1][7]=a9;
  a9=(a13*a17);
  a5=(a11*a16);
  a9=(a9+a5);
  a9=(-a9);
  if (res[1]!=0) res[1][8]=a9;
  a9=(a12*a17);
  a5=(a21*a16);
  a9=(a9+a5);
  if (res[1]!=0) res[1][9]=a9;
  a17=(a10*a17);
  a16=(a15*a16);
  a17=(a17+a16);
  if (res[1]!=0) res[1][10]=a17;
  a17=0.;
  if (res[1]!=0) res[1][11]=a17;
  if (res[1]!=0) res[1][12]=a17;
  if (res[1]!=0) res[1][13]=a17;
  if (res[1]!=0) res[1][14]=a17;
  if (res[1]!=0) res[1][15]=a17;
  if (res[1]!=0) res[1][16]=a17;
  if (res[1]!=0) res[1][17]=a17;
  if (res[1]!=0) res[1][18]=a17;
  if (res[1]!=0) res[1][19]=a17;
  if (res[1]!=0) res[1][20]=a17;
  a17=8.;
  if (res[2]!=0) res[2][0]=a17;
  if (res[2]!=0) res[2][1]=a17;
  if (res[2]!=0) res[2][2]=a17;
  if (res[2]!=0) res[2][3]=a17;
  a17=300.;
  if (res[2]!=0) res[2][4]=a17;
  if (res[2]!=0) res[2][5]=a17;
  a17=5400.;
  if (res[2]!=0) res[2][6]=a17;
  a17=(a8*a19);
  a17=(a14*a17);
  a16=(a8*a14);
  a16=(a19*a16);
  a17=(a17+a16);
  if (res[2]!=0) res[2][7]=a17;
  a17=(a8*a11);
  a16=(a14*a17);
  a9=(a8*a13);
  a5=(a19*a9);
  a16=(a16+a5);
  if (res[2]!=0) res[2][8]=a16;
  a5=(a8*a21);
  a7=(a14*a5);
  a6=(a8*a12);
  a0=(a19*a6);
  a7=(a7+a0);
  a7=(-a7);
  if (res[2]!=0) res[2][9]=a7;
  a0=(a8*a15);
  a14=(a14*a0);
  a8=(a8*a10);
  a19=(a19*a8);
  a14=(a14+a19);
  a14=(-a14);
  if (res[2]!=0) res[2][10]=a14;
  if (res[2]!=0) res[2][11]=a16;
  a17=(a13*a17);
  a9=(a11*a9);
  a17=(a17+a9);
  if (res[2]!=0) res[2][12]=a17;
  a17=(a13*a5);
  a9=(a11*a6);
  a17=(a17+a9);
  a17=(-a17);
  if (res[2]!=0) res[2][13]=a17;
  a13=(a13*a0);
  a11=(a11*a8);
  a13=(a13+a11);
  a13=(-a13);
  if (res[2]!=0) res[2][14]=a13;
  if (res[2]!=0) res[2][15]=a7;
  if (res[2]!=0) res[2][16]=a17;
  a5=(a12*a5);
  a6=(a21*a6);
  a5=(a5+a6);
  if (res[2]!=0) res[2][17]=a5;
  a12=(a12*a0);
  a21=(a21*a8);
  a12=(a12+a21);
  if (res[2]!=0) res[2][18]=a12;
  if (res[2]!=0) res[2][19]=a14;
  if (res[2]!=0) res[2][20]=a13;
  if (res[2]!=0) res[2][21]=a12;
  a10=(a10*a0);
  a15=(a15*a8);
  a10=(a10+a15);
  if (res[2]!=0) res[2][22]=a10;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_0_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_0_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_cost_ext_cost_0_fun_jac_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_0_fun_jac_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_0_fun_jac_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_0_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_0_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    case 3: return casadi_s2;
    case 4: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_hess_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 5*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
