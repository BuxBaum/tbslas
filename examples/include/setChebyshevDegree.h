#include "preprocessordefines.h"

// Edit the values here to change the Chebyshev degree used for evaluation
// then recompile


// Chebyshev degree to evaluate with +1 (parameter q+1)
// D up to 17 are possible (q 16)
#define D 15

// compute with d*(d+1)*(d+2)/6
// see list below for values
#define NUMCOEFF 680

// set according to D
#define COMP() COMP15()
#define COMP_1() COMP15_1()



/*
 * NUMCOEFF LIST
 * 
 * d 			NUMCOEFF
 * 1				1
 * 2				4
 * 3				10
 * 4				20
 * 5				35
 * 6				56
 * 7				84
 * 8				120
 * 9				165
 * 10				220
 * 11				286
 * 12				364
 * 13				455
 * 14				560
 * 15				680
 * 16				816
 * 17				969
 */ 
