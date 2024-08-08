/* This file was automatically generated by CasADi 3.6.4.
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
  #define CASADI_PREFIX(ID) drone_ode_cost_ext_cost_fun_ ## ID
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
static const casadi_int casadi_s4[5] = {1, 1, 0, 1, 0};

/* drone_ode_cost_ext_cost_fun:(i0[13],i1[4],i2[],i3[27])->(o0) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=300.;
  a1=arg[0]? arg[0][0] : 0;
  a2=arg[3]? arg[3][14] : 0;
  a3=(a1-a2);
  a3=(a0*a3);
  a1=(a1-a2);
  a3=(a3*a1);
  a1=arg[0]? arg[0][1] : 0;
  a2=arg[3]? arg[3][15] : 0;
  a4=(a1-a2);
  a0=(a0*a4);
  a1=(a1-a2);
  a0=(a0*a1);
  a3=(a3+a0);
  a0=700.;
  a1=arg[0]? arg[0][2] : 0;
  a2=arg[3]? arg[3][16] : 0;
  a4=(a1-a2);
  a0=(a0*a4);
  a1=(a1-a2);
  a0=(a0*a1);
  a3=(a3+a0);
  a0=80.;
  a1=arg[3]? arg[3][17] : 0;
  a2=casadi_sq(a1);
  a4=arg[3]? arg[3][18] : 0;
  a5=casadi_sq(a4);
  a2=(a2+a5);
  a5=arg[3]? arg[3][19] : 0;
  a6=casadi_sq(a5);
  a2=(a2+a6);
  a6=arg[3]? arg[3][20] : 0;
  a7=casadi_sq(a6);
  a2=(a2+a7);
  a7=(a1/a2);
  a8=arg[0]? arg[0][6] : 0;
  a7=(a7*a8);
  a9=arg[0]? arg[0][3] : 0;
  a10=(a6/a2);
  a10=(a9*a10);
  a7=(a7-a10);
  a10=arg[0]? arg[0][5] : 0;
  a11=(a4/a2);
  a11=(a10*a11);
  a12=arg[0]? arg[0][4] : 0;
  a2=(a5/a2);
  a2=(a12*a2);
  a11=(a11-a2);
  a7=(a7+a11);
  a0=(a0*a7);
  a7=casadi_sq(a1);
  a11=casadi_sq(a4);
  a7=(a7+a11);
  a11=casadi_sq(a5);
  a7=(a7+a11);
  a11=casadi_sq(a6);
  a7=(a7+a11);
  a1=(a1/a7);
  a1=(a1*a8);
  a6=(a6/a7);
  a9=(a9*a6);
  a1=(a1-a9);
  a4=(a4/a7);
  a10=(a10*a4);
  a5=(a5/a7);
  a12=(a12*a5);
  a10=(a10-a12);
  a1=(a1+a10);
  a0=(a0*a1);
  a3=(a3+a0);
  a0=2.0000000000000001e-01;
  a1=arg[1]? arg[1][0] : 0;
  a10=(a0*a1);
  a10=(a10*a1);
  a1=arg[1]? arg[1][1] : 0;
  a12=(a0*a1);
  a12=(a12*a1);
  a10=(a10+a12);
  a12=arg[1]? arg[1][2] : 0;
  a1=(a0*a12);
  a1=(a1*a12);
  a10=(a10+a1);
  a1=arg[1]? arg[1][3] : 0;
  a0=(a0*a1);
  a0=(a0*a1);
  a10=(a10+a0);
  a3=(a3+a10);
  if (res[0]!=0) res[0][0]=a3;
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void drone_ode_cost_ext_cost_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_fun_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int drone_ode_cost_ext_cost_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real drone_ode_cost_ext_cost_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* drone_ode_cost_ext_cost_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* drone_ode_cost_ext_cost_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int drone_ode_cost_ext_cost_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
