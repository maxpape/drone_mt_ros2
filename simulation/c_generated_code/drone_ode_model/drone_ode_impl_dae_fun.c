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
  #define CASADI_PREFIX(ID) drone_ode_impl_dae_fun_ ## ID
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

/* drone_ode_impl_dae_fun:(i0[17],i1[17],i2[4],i3[],i4[],i5[21])->(o0[17]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a3, a4, a5, a6, a7, a8, a9;
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
  a1=arg[0]? arg[0][4] : 0;
  a2=arg[0]? arg[0][10] : 0;
  a3=(a1*a2);
  a4=arg[0]? arg[0][5] : 0;
  a5=arg[0]? arg[0][11] : 0;
  a6=(a4*a5);
  a3=(a3+a6);
  a6=arg[0]? arg[0][6] : 0;
  a7=arg[0]? arg[0][12] : 0;
  a8=(a6*a7);
  a3=(a3+a8);
  a8=2.;
  a3=(a3/a8);
  a0=(a0+a3);
  if (res[0]!=0) res[0][3]=a0;
  a0=arg[1]? arg[1][4] : 0;
  a3=arg[0]? arg[0][3] : 0;
  a9=(a3*a2);
  a10=(a4*a7);
  a11=(a6*a5);
  a10=(a10-a11);
  a9=(a9+a10);
  a9=(a9/a8);
  a0=(a0-a9);
  if (res[0]!=0) res[0][4]=a0;
  a0=arg[1]? arg[1][5] : 0;
  a9=(a3*a5);
  a10=(a6*a2);
  a11=(a1*a7);
  a10=(a10-a11);
  a9=(a9+a10);
  a9=(a9/a8);
  a0=(a0-a9);
  if (res[0]!=0) res[0][5]=a0;
  a0=arg[1]? arg[1][6] : 0;
  a9=(a3*a7);
  a10=(a1*a5);
  a11=(a4*a2);
  a10=(a10-a11);
  a9=(a9+a10);
  a9=(a9/a8);
  a0=(a0-a9);
  if (res[0]!=0) res[0][6]=a0;
  a0=arg[1]? arg[1][7] : 0;
  a9=arg[2]? arg[2][0] : 0;
  a8=arg[2]? arg[2][1] : 0;
  a10=(a9+a8);
  a11=arg[2]? arg[2][2] : 0;
  a10=(a10+a11);
  a12=arg[2]? arg[2][3] : 0;
  a10=(a10+a12);
  a13=(a6*a10);
  a14=casadi_sq(a3);
  a15=casadi_sq(a1);
  a14=(a14+a15);
  a15=casadi_sq(a4);
  a14=(a14+a15);
  a15=casadi_sq(a6);
  a14=(a14+a15);
  a15=(a1/a14);
  a16=(a13*a15);
  a17=(a3/a14);
  a18=(a4*a10);
  a19=(a17*a18);
  a16=(a16+a19);
  a1=(a1*a10);
  a6=(a6/a14);
  a19=(a1*a6);
  a3=(a3*a10);
  a4=(a4/a14);
  a14=(a3*a4);
  a19=(a19+a14);
  a16=(a16+a19);
  a19=arg[5]? arg[5][0] : 0;
  a16=(a16/a19);
  a0=(a0-a16);
  if (res[0]!=0) res[0][7]=a0;
  a0=arg[1]? arg[1][8] : 0;
  a16=(a13*a4);
  a14=(a17*a1);
  a16=(a16-a14);
  a14=(a18*a6);
  a10=(a3*a15);
  a14=(a14-a10);
  a16=(a16+a14);
  a16=(a16/a19);
  a0=(a0-a16);
  if (res[0]!=0) res[0][8]=a0;
  a0=arg[1]? arg[1][9] : 0;
  a13=(a13*a6);
  a17=(a17*a3);
  a13=(a13+a17);
  a18=(a18*a4);
  a1=(a1*a15);
  a18=(a18+a1);
  a13=(a13-a18);
  a13=(a13/a19);
  a19=arg[5]? arg[5][1] : 0;
  a13=(a13+a19);
  a0=(a0-a13);
  if (res[0]!=0) res[0][9]=a0;
  a0=arg[1]? arg[1][10] : 0;
  a13=arg[5]? arg[5][7] : 0;
  a13=(a13*a11);
  a19=arg[5]? arg[5][5] : 0;
  a19=(a19*a9);
  a18=arg[5]? arg[5][6] : 0;
  a18=(a18*a8);
  a19=(a19+a18);
  a13=(a13-a19);
  a19=arg[5]? arg[5][8] : 0;
  a19=(a19*a12);
  a13=(a13+a19);
  a19=arg[5]? arg[5][4] : 0;
  a18=(a19*a7);
  a1=(a5*a18);
  a15=arg[5]? arg[5][3] : 0;
  a4=(a15*a5);
  a17=(a7*a4);
  a1=(a1-a17);
  a13=(a13-a1);
  a1=arg[5]? arg[5][2] : 0;
  a13=(a13/a1);
  a0=(a0-a13);
  if (res[0]!=0) res[0][10]=a0;
  a0=arg[1]? arg[1][11] : 0;
  a13=arg[5]? arg[5][9] : 0;
  a13=(a13*a9);
  a17=arg[5]? arg[5][10] : 0;
  a17=(a17*a8);
  a13=(a13-a17);
  a17=arg[5]? arg[5][11] : 0;
  a17=(a17*a11);
  a13=(a13-a17);
  a17=arg[5]? arg[5][12] : 0;
  a17=(a17*a12);
  a13=(a13+a17);
  a1=(a1*a2);
  a7=(a7*a1);
  a18=(a2*a18);
  a7=(a7-a18);
  a13=(a13-a7);
  a13=(a13/a15);
  a0=(a0-a13);
  if (res[0]!=0) res[0][11]=a0;
  a0=arg[1]? arg[1][12] : 0;
  a13=arg[5]? arg[5][13] : 0;
  a15=(a13*a8);
  a7=(a13*a9);
  a15=(a15-a7);
  a7=(a13*a11);
  a15=(a15-a7);
  a13=(a13*a12);
  a15=(a15+a13);
  a2=(a2*a4);
  a5=(a5*a1);
  a2=(a2-a5);
  a15=(a15-a2);
  a15=(a15/a19);
  a0=(a0-a15);
  if (res[0]!=0) res[0][12]=a0;
  a0=arg[1]? arg[1][13] : 0;
  a15=10.;
  a19=arg[0]? arg[0][13] : 0;
  a9=(a9-a19);
  a9=(a15*a9);
  a0=(a0-a9);
  if (res[0]!=0) res[0][13]=a0;
  a0=arg[1]? arg[1][14] : 0;
  a9=arg[0]? arg[0][14] : 0;
  a8=(a8-a9);
  a8=(a15*a8);
  a0=(a0-a8);
  if (res[0]!=0) res[0][14]=a0;
  a0=arg[1]? arg[1][15] : 0;
  a8=arg[0]? arg[0][15] : 0;
  a11=(a11-a8);
  a11=(a15*a11);
  a0=(a0-a11);
  if (res[0]!=0) res[0][15]=a0;
  a0=arg[1]? arg[1][16] : 0;
  a11=arg[0]? arg[0][16] : 0;
  a12=(a12-a11);
  a15=(a15*a12);
  a0=(a0-a15);
  if (res[0]!=0) res[0][16]=a0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_impl_dae_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_impl_dae_fun_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_impl_dae_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_impl_dae_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_impl_dae_fun_name_in(casadi_int i) {
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

CASADI_SYMBOL_EXPORT const char* drone_ode_impl_dae_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_impl_dae_fun_sparsity_in(casadi_int i) {
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

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_impl_dae_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_impl_dae_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
