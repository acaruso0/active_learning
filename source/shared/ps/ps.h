#ifndef CU_INCLUDE_POTENTIAL_1B_PS_H
#define CU_INCLUDE_POTENTIAL_1B_PS_H

#include <algorithm>
#include <vector>

/**
 * @file ps.h
 * @brief This file contains the calls for the energy of water
 */

/**
 * @namespace ps
 * @brief Namespace of the water energy functions
 */
namespace ps {

/**
 * @brief Computes the energy and gradients for a water monomer
 *
 * Given the coordinates of a water monomer, it returns the 1b energy of
 * that water monomer.
 * @param[in] rr Coordinates of the water molecule (OHH)
 * @param[out] dr Gradients of the atoms of the water molecule (OxyzHxyzHxyz)
 * @return Deformation energy of the water molecule
 */
// double pot_nasa(const double* rr, double* dr);

/**
 * @brief Computes the energy and gradients for a water monomer
 *
 * Given the coordinates of a water monomer, it returns the 1b energy of
 * all the water molecules in a vector.
 * @param[in] rr Coordinates of the all the water molecule
 * @param[out] dr Gradients of all the water molecules
 * @param[in] nw Number of water molecules
 * @return Deformation energy of all the water molecules in a vector
 */
std::vector<double> pot_nasa(const std::vector<double> rr, size_t nw);

}  // namespace ps

#endif  // CU_INCLUDE_POTENTIAL_1B_PS_H
