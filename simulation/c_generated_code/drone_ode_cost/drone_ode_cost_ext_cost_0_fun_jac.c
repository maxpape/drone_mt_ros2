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
  #define CASADI_PREFIX(ID) drone_ode_cost_ext_cost_0_fun_jac_ ## ID
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

static const casadi_int casadi_s0[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s1[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s2[3] = {0, 0, 0};
static const casadi_int casadi_s3[25] = {21, 1, 0, 21, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s5[19] = {15, 1, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

/* drone_ode_cost_ext_cost_0_fun_jac:(i0[11],i1[4],i2[],i3[21])->(o0,o1[15]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a5, a6, a7, a8, a9;
  a0=2.;
  a1=arg[0]? arg[0][0] : 0;
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][1] : 0;
  a4=casadi_sq(a3);
  a2=(a2+a4);
  a4=arg[0]? arg[0][2] : 0;
  a5=casadi_sq(a4);
  a2=(a2+a5);
  a5=arg[0]? arg[0][3] : 0;
  a6=casadi_sq(a5);
  a2=(a2+a6);
  a6=(a1/a2);
  a7=arg[3]? arg[3][15] : 0;
  a8=(a6*a7);
  a9=arg[3]? arg[3][14] : 0;
  a10=(a3/a2);
  a11=(a9*a10);
  a8=(a8-a11);
  a11=arg[3]? arg[3][17] : 0;
  a12=(a4/a2);
  a13=(a11*a12);
  a14=arg[3]? arg[3][16] : 0;
  a15=(a5/a2);
  a16=(a14*a15);
  a13=(a13-a16);
  a8=(a8+a13);
  a8=(a0*a8);
  a13=casadi_sq(a1);
  a16=casadi_sq(a3);
  a13=(a13+a16);
  a16=casadi_sq(a4);
  a13=(a13+a16);
  a16=casadi_sq(a5);
  a13=(a13+a16);
  a16=(a1/a13);
  a17=(a16*a7);
  a18=(a3/a13);
  a19=(a9*a18);
  a17=(a17-a19);
  a19=(a4/a13);
  a20=(a11*a19);
  a21=(a5/a13);
  a22=(a14*a21);
  a20=(a20-a22);
  a17=(a17+a20);
  a20=(a8*a17);
  a22=(a6*a14);
  a23=(a9*a12);
  a22=(a22-a23);
  a23=(a7*a15);
  a24=(a11*a10);
  a23=(a23-a24);
  a22=(a22+a23);
  a22=(a0*a22);
  a23=(a16*a14);
  a24=(a9*a19);
  a23=(a23-a24);
  a24=(a7*a21);
  a25=(a11*a18);
  a24=(a24-a25);
  a23=(a23+a24);
  a24=(a22*a23);
  a20=(a20+a24);
  a24=(a6*a11);
  a25=(a9*a15);
  a24=(a24-a25);
  a25=(a14*a10);
  a26=(a7*a12);
  a25=(a25-a26);
  a24=(a24+a25);
  a24=(a0*a24);
  a25=(a16*a11);
  a26=(a9*a21);
  a25=(a25-a26);
  a26=(a14*a18);
  a27=(a7*a19);
  a26=(a26-a27);
  a25=(a25+a26);
  a26=(a24*a25);
  a20=(a20+a26);
  a26=arg[3]? arg[3][18] : 0;
  a27=arg[0]? arg[0][4] : 0;
  a28=(a26-a27);
  a28=(a0*a28);
  a26=(a26-a27);
  a27=(a28*a26);
  a20=(a20+a27);
  a27=1.0000000000000001e-01;
  a29=arg[3]? arg[3][19] : 0;
  a30=arg[0]? arg[0][5] : 0;
  a31=(a29-a30);
  a31=(a27*a31);
  a29=(a29-a30);
  a30=(a31*a29);
  a20=(a20+a30);
  a30=arg[3]? arg[3][20] : 0;
  a32=arg[0]? arg[0][6] : 0;
  a33=(a30-a32);
  a33=(a27*a33);
  a30=(a30-a32);
  a32=(a33*a30);
  a20=(a20+a32);
  a32=arg[1]? arg[1][0] : 0;
  a34=(a27*a32);
  a35=(a34*a32);
  a36=arg[1]? arg[1][1] : 0;
  a37=(a27*a36);
  a38=(a37*a36);
  a35=(a35+a38);
  a38=arg[1]? arg[1][2] : 0;
  a39=(a27*a38);
  a40=(a39*a38);
  a35=(a35+a40);
  a40=arg[1]? arg[1][3] : 0;
  a41=(a27*a40);
  a42=(a41*a40);
  a35=(a35+a42);
  a20=(a20+a35);
  if (res[0]!=0) res[0][0]=a20;
  a32=(a27*a32);
  a34=(a34+a32);
  if (res[1]!=0) res[1][0]=a34;
  a36=(a27*a36);
  a37=(a37+a36);
  if (res[1]!=0) res[1][1]=a37;
  a38=(a27*a38);
  a39=(a39+a38);
  if (res[1]!=0) res[1][2]=a39;
  a40=(a27*a40);
  a41=(a41+a40);
  if (res[1]!=0) res[1][3]=a41;
  a41=(a11*a24);
  a40=(a14*a22);
  a41=(a41+a40);
  a40=(a7*a8);
  a41=(a41+a40);
  a40=(a41/a13);
  a39=(a1+a1);
  a21=(a21/a13);
  a38=(a7*a22);
  a37=(a9*a24);
  a38=(a38-a37);
  a37=(a14*a8);
  a38=(a38-a37);
  a21=(a21*a38);
  a19=(a19/a13);
  a37=(a11*a8);
  a36=(a7*a24);
  a34=(a9*a22);
  a36=(a36+a34);
  a37=(a37-a36);
  a19=(a19*a37);
  a21=(a21+a19);
  a18=(a18/a13);
  a24=(a14*a24);
  a22=(a11*a22);
  a24=(a24-a22);
  a8=(a9*a8);
  a24=(a24-a8);
  a18=(a18*a24);
  a21=(a21+a18);
  a16=(a16/a13);
  a16=(a16*a41);
  a21=(a21+a16);
  a39=(a39*a21);
  a40=(a40-a39);
  a25=(a0*a25);
  a39=(a11*a25);
  a23=(a0*a23);
  a16=(a14*a23);
  a39=(a39+a16);
  a17=(a0*a17);
  a16=(a7*a17);
  a39=(a39+a16);
  a16=(a39/a2);
  a40=(a40+a16);
  a1=(a1+a1);
  a15=(a15/a2);
  a16=(a7*a23);
  a41=(a9*a25);
  a16=(a16-a41);
  a41=(a14*a17);
  a16=(a16-a41);
  a15=(a15*a16);
  a12=(a12/a2);
  a41=(a11*a17);
  a7=(a7*a25);
  a18=(a9*a23);
  a7=(a7+a18);
  a41=(a41-a7);
  a12=(a12*a41);
  a15=(a15+a12);
  a10=(a10/a2);
  a14=(a14*a25);
  a11=(a11*a23);
  a14=(a14-a11);
  a9=(a9*a17);
  a14=(a14-a9);
  a10=(a10*a14);
  a15=(a15+a10);
  a6=(a6/a2);
  a6=(a6*a39);
  a15=(a15+a6);
  a1=(a1*a15);
  a40=(a40-a1);
  if (res[1]!=0) res[1][4]=a40;
  a24=(a24/a13);
  a40=(a3+a3);
  a40=(a40*a21);
  a24=(a24-a40);
  a14=(a14/a2);
  a24=(a24+a14);
  a3=(a3+a3);
  a3=(a3*a15);
  a24=(a24-a3);
  if (res[1]!=0) res[1][5]=a24;
  a37=(a37/a13);
  a24=(a4+a4);
  a24=(a24*a21);
  a37=(a37-a24);
  a41=(a41/a2);
  a37=(a37+a41);
  a4=(a4+a4);
  a4=(a4*a15);
  a37=(a37-a4);
  if (res[1]!=0) res[1][6]=a37;
  a38=(a38/a13);
  a13=(a5+a5);
  a13=(a13*a21);
  a38=(a38-a13);
  a16=(a16/a2);
  a38=(a38+a16);
  a5=(a5+a5);
  a5=(a5*a15);
  a38=(a38-a5);
  if (res[1]!=0) res[1][7]=a38;
  a0=(a0*a26);
  a28=(a28+a0);
  a28=(-a28);
  if (res[1]!=0) res[1][8]=a28;
  a29=(a27*a29);
  a31=(a31+a29);
  a31=(-a31);
  if (res[1]!=0) res[1][9]=a31;
  a27=(a27*a30);
  a33=(a33+a27);
  a33=(-a33);
  if (res[1]!=0) res[1][10]=a33;
  a33=0.;
  if (res[1]!=0) res[1][11]=a33;
  if (res[1]!=0) res[1][12]=a33;
  if (res[1]!=0) res[1][13]=a33;
  if (res[1]!=0) res[1][14]=a33;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_0_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_0_fun_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_0_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_cost_ext_cost_0_fun_jac_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_0_fun_jac_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_0_fun_jac_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_0_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_0_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_0_fun_jac_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
