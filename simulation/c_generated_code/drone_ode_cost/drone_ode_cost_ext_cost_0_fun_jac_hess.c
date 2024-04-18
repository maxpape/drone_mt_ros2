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
static const casadi_int casadi_s3[25] = {21, 1, 0, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s5[47] = {21, 21, 0, 1, 2, 3, 4, 5, 6, 7, 11, 15, 19, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10};
static const casadi_int casadi_s6[24] = {0, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* drone_ode_cost_ext_cost_0_fun_jac_hess:(i0[17],i1[4],i2[],i3[21])->(o0,o1[21],o2[21x21,23nz],o3[],o4[0x21]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][0] : 0;
  a1=arg[3]? arg[3][14] : 0;
  a0=(a0-a1);
  a1=casadi_sq(a0);
  a2=arg[0]? arg[0][1] : 0;
  a3=arg[3]? arg[3][15] : 0;
  a2=(a2-a3);
  a3=casadi_sq(a2);
  a1=(a1+a3);
  a3=1000.;
  a4=arg[0]? arg[0][2] : 0;
  a5=arg[3]? arg[3][16] : 0;
  a6=(a4-a5);
  a6=(a3*a6);
  a4=(a4-a5);
  a5=(a6*a4);
  a1=(a1+a5);
  a5=1.0000000000000000e-03;
  a7=arg[3]? arg[3][17] : 0;
  a8=casadi_sq(a7);
  a9=arg[3]? arg[3][18] : 0;
  a10=casadi_sq(a9);
  a8=(a8+a10);
  a10=arg[3]? arg[3][19] : 0;
  a11=casadi_sq(a10);
  a8=(a8+a11);
  a11=arg[3]? arg[3][20] : 0;
  a12=casadi_sq(a11);
  a8=(a8+a12);
  a12=(a7/a8);
  a13=arg[0]? arg[0][4] : 0;
  a14=(a12*a13);
  a15=arg[0]? arg[0][3] : 0;
  a16=(a9/a8);
  a17=(a15*a16);
  a14=(a14-a17);
  a17=arg[0]? arg[0][6] : 0;
  a18=(a10/a8);
  a19=(a17*a18);
  a20=arg[0]? arg[0][5] : 0;
  a8=(a11/a8);
  a21=(a20*a8);
  a19=(a19-a21);
  a14=(a14+a19);
  a14=(a5*a14);
  a19=casadi_sq(a7);
  a21=casadi_sq(a9);
  a19=(a19+a21);
  a21=casadi_sq(a10);
  a19=(a19+a21);
  a21=casadi_sq(a11);
  a19=(a19+a21);
  a7=(a7/a19);
  a21=(a7*a13);
  a9=(a9/a19);
  a22=(a15*a9);
  a21=(a21-a22);
  a10=(a10/a19);
  a22=(a17*a10);
  a11=(a11/a19);
  a19=(a20*a11);
  a22=(a22-a19);
  a21=(a21+a22);
  a22=(a14*a21);
  a1=(a1+a22);
  a22=(a12*a20);
  a19=(a15*a18);
  a22=(a22-a19);
  a19=(a13*a8);
  a23=(a17*a16);
  a19=(a19-a23);
  a22=(a22+a19);
  a22=(a5*a22);
  a19=(a7*a20);
  a23=(a15*a10);
  a19=(a19-a23);
  a23=(a13*a11);
  a24=(a17*a9);
  a23=(a23-a24);
  a19=(a19+a23);
  a23=(a22*a19);
  a1=(a1+a23);
  a23=3.0000000000000001e-03;
  a24=(a12*a17);
  a25=(a15*a8);
  a24=(a24-a25);
  a25=(a20*a16);
  a26=(a13*a18);
  a25=(a25-a26);
  a24=(a24+a25);
  a24=(a23*a24);
  a17=(a7*a17);
  a15=(a15*a11);
  a17=(a17-a15);
  a20=(a20*a9);
  a13=(a13*a10);
  a20=(a20-a13);
  a17=(a17+a20);
  a20=(a24*a17);
  a1=(a1+a20);
  a20=arg[1]? arg[1][0] : 0;
  a13=casadi_sq(a20);
  a15=arg[1]? arg[1][1] : 0;
  a25=casadi_sq(a15);
  a13=(a13+a25);
  a25=arg[1]? arg[1][2] : 0;
  a26=casadi_sq(a25);
  a13=(a13+a26);
  a26=arg[1]? arg[1][3] : 0;
  a27=casadi_sq(a26);
  a13=(a13+a27);
  a1=(a1+a13);
  if (res[0]!=0) res[0][0]=a1;
  a20=(a20+a20);
  if (res[1]!=0) res[1][0]=a20;
  a15=(a15+a15);
  if (res[1]!=0) res[1][1]=a15;
  a25=(a25+a25);
  if (res[1]!=0) res[1][2]=a25;
  a26=(a26+a26);
  if (res[1]!=0) res[1][3]=a26;
  a0=(a0+a0);
  if (res[1]!=0) res[1][4]=a0;
  a2=(a2+a2);
  if (res[1]!=0) res[1][5]=a2;
  a3=(a3*a4);
  a6=(a6+a3);
  if (res[1]!=0) res[1][6]=a6;
  a6=(a11*a24);
  a17=(a23*a17);
  a3=(a8*a17);
  a6=(a6+a3);
  a3=(a10*a22);
  a6=(a6+a3);
  a19=(a5*a19);
  a3=(a18*a19);
  a6=(a6+a3);
  a3=(a9*a14);
  a6=(a6+a3);
  a21=(a5*a21);
  a3=(a16*a21);
  a6=(a6+a3);
  a6=(-a6);
  if (res[1]!=0) res[1][7]=a6;
  a6=(a11*a22);
  a3=(a10*a24);
  a4=(a18*a17);
  a3=(a3+a4);
  a6=(a6-a3);
  a3=(a8*a19);
  a6=(a6+a3);
  a3=(a7*a14);
  a6=(a6+a3);
  a3=(a12*a21);
  a6=(a6+a3);
  if (res[1]!=0) res[1][8]=a6;
  a6=(a9*a24);
  a3=(a16*a17);
  a6=(a6+a3);
  a3=(a7*a22);
  a6=(a6+a3);
  a3=(a12*a19);
  a6=(a6+a3);
  a3=(a11*a14);
  a6=(a6-a3);
  a3=(a8*a21);
  a6=(a6-a3);
  if (res[1]!=0) res[1][9]=a6;
  a24=(a7*a24);
  a17=(a12*a17);
  a24=(a24+a17);
  a22=(a9*a22);
  a24=(a24-a22);
  a19=(a16*a19);
  a24=(a24-a19);
  a14=(a10*a14);
  a24=(a24+a14);
  a21=(a18*a21);
  a24=(a24+a21);
  if (res[1]!=0) res[1][10]=a24;
  a24=0.;
  if (res[1]!=0) res[1][11]=a24;
  if (res[1]!=0) res[1][12]=a24;
  if (res[1]!=0) res[1][13]=a24;
  if (res[1]!=0) res[1][14]=a24;
  if (res[1]!=0) res[1][15]=a24;
  if (res[1]!=0) res[1][16]=a24;
  if (res[1]!=0) res[1][17]=a24;
  if (res[1]!=0) res[1][18]=a24;
  if (res[1]!=0) res[1][19]=a24;
  if (res[1]!=0) res[1][20]=a24;
  a24=2.;
  if (res[2]!=0) res[2][0]=a24;
  if (res[2]!=0) res[2][1]=a24;
  if (res[2]!=0) res[2][2]=a24;
  if (res[2]!=0) res[2][3]=a24;
  if (res[2]!=0) res[2][4]=a24;
  if (res[2]!=0) res[2][5]=a24;
  a24=2000.;
  if (res[2]!=0) res[2][6]=a24;
  a24=(a23*a8);
  a24=(a11*a24);
  a21=(a23*a11);
  a21=(a8*a21);
  a24=(a24+a21);
  a21=(a5*a18);
  a21=(a10*a21);
  a24=(a24+a21);
  a21=(a5*a10);
  a21=(a18*a21);
  a24=(a24+a21);
  a21=(a5*a16);
  a21=(a9*a21);
  a24=(a24+a21);
  a21=(a5*a9);
  a21=(a16*a21);
  a24=(a24+a21);
  if (res[2]!=0) res[2][7]=a24;
  a24=(a5*a8);
  a21=(a10*a24);
  a14=(a23*a18);
  a19=(a11*a14);
  a22=(a23*a10);
  a17=(a8*a22);
  a19=(a19+a17);
  a21=(a21-a19);
  a19=(a5*a11);
  a17=(a18*a19);
  a21=(a21+a17);
  a17=(a5*a12);
  a6=(a9*a17);
  a21=(a21+a6);
  a6=(a5*a7);
  a3=(a16*a6);
  a21=(a21+a3);
  a21=(-a21);
  if (res[2]!=0) res[2][8]=a21;
  a3=(a23*a16);
  a4=(a11*a3);
  a2=(a23*a9);
  a0=(a8*a2);
  a4=(a4+a0);
  a0=(a5*a12);
  a26=(a10*a0);
  a4=(a4+a26);
  a26=(a5*a7);
  a25=(a18*a26);
  a4=(a4+a25);
  a25=(a5*a8);
  a15=(a9*a25);
  a4=(a4-a15);
  a15=(a5*a11);
  a20=(a16*a15);
  a4=(a4-a20);
  a4=(-a4);
  if (res[2]!=0) res[2][9]=a4;
  a20=(a23*a12);
  a1=(a11*a20);
  a23=(a23*a7);
  a13=(a8*a23);
  a1=(a1+a13);
  a13=(a5*a16);
  a27=(a10*a13);
  a1=(a1-a27);
  a27=(a5*a9);
  a28=(a18*a27);
  a1=(a1-a28);
  a28=(a5*a18);
  a29=(a9*a28);
  a1=(a1+a29);
  a5=(a5*a10);
  a29=(a16*a5);
  a1=(a1+a29);
  a1=(-a1);
  if (res[2]!=0) res[2][10]=a1;
  if (res[2]!=0) res[2][11]=a21;
  a24=(a11*a24);
  a14=(a10*a14);
  a22=(a18*a22);
  a14=(a14+a22);
  a24=(a24+a14);
  a19=(a8*a19);
  a24=(a24+a19);
  a17=(a7*a17);
  a24=(a24+a17);
  a6=(a12*a6);
  a24=(a24+a6);
  if (res[2]!=0) res[2][12]=a24;
  a24=(a11*a0);
  a6=(a10*a3);
  a17=(a18*a2);
  a6=(a6+a17);
  a24=(a24-a6);
  a6=(a8*a26);
  a24=(a24+a6);
  a6=(a7*a25);
  a24=(a24-a6);
  a6=(a12*a15);
  a24=(a24-a6);
  if (res[2]!=0) res[2][13]=a24;
  a6=(a7*a28);
  a17=(a11*a13);
  a19=(a10*a20);
  a14=(a18*a23);
  a19=(a19+a14);
  a17=(a17+a19);
  a19=(a8*a27);
  a17=(a17+a19);
  a6=(a6-a17);
  a17=(a12*a5);
  a6=(a6+a17);
  if (res[2]!=0) res[2][14]=a6;
  if (res[2]!=0) res[2][15]=a4;
  if (res[2]!=0) res[2][16]=a24;
  a3=(a9*a3);
  a2=(a16*a2);
  a3=(a3+a2);
  a0=(a7*a0);
  a3=(a3+a0);
  a26=(a12*a26);
  a3=(a3+a26);
  a25=(a11*a25);
  a3=(a3+a25);
  a15=(a8*a15);
  a3=(a3+a15);
  if (res[2]!=0) res[2][17]=a3;
  a3=(a9*a20);
  a15=(a16*a23);
  a3=(a3+a15);
  a15=(a7*a13);
  a3=(a3-a15);
  a15=(a12*a27);
  a3=(a3-a15);
  a11=(a11*a28);
  a3=(a3-a11);
  a8=(a8*a5);
  a3=(a3-a8);
  if (res[2]!=0) res[2][18]=a3;
  if (res[2]!=0) res[2][19]=a1;
  if (res[2]!=0) res[2][20]=a6;
  if (res[2]!=0) res[2][21]=a3;
  a7=(a7*a20);
  a12=(a12*a23);
  a7=(a7+a12);
  a9=(a9*a13);
  a7=(a7+a9);
  a16=(a16*a27);
  a7=(a7+a16);
  a10=(a10*a28);
  a7=(a7+a10);
  a18=(a18*a5);
  a7=(a7+a18);
  if (res[2]!=0) res[2][22]=a7;
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
    case 1: return casadi_s3;
    case 2: return casadi_s5;
    case 3: return casadi_s2;
    case 4: return casadi_s6;
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