/* iPIC3D was originally developed by Stefano Markidis and Giovanni Lapenta. 
 * This release was contributed by Alec Johnson and Ivy Bo Peng.
 * Publications that use results from iPIC3D need to properly cite  
 * 'S. Markidis, G. Lapenta, and Rizwan-uddin. "Multi-scale simulations of 
 * plasma with iPIC3D." Mathematics and Computers in Simulation 80.7 (2010): 1509-1519.'
 *
 *        Copyright 2015 KTH Royal Institute of Technology
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at 
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mpi.h>
#include <fstream>

#include "ParallelIO.h"
#include "MPIdata.h"
#include "TimeTasks.h"
#include "Collective.h"
#include "Grid3DCU.h"
#include "VCtopology3D.h"
#include "Particles3D.h"
#include "EMfields3D.h"
#include "math.h"
#include <algorithm>
#include <iostream>
#include <sstream>



// template<typename T, int sz>
// int size(T(&)[sz])
// {
//     return sz;
// }

void WriteFieldsVTK(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, const string & outputTag ,int cycle){

	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	MPI_File     fh;
	MPI_Status   status;

	if (outputTag.find("B", 0) != string::npos || outputTag.find("E", 0) != string::npos
			 || outputTag.find("Je", 0) != string::npos || outputTag.find("Ji", 0) != string::npos){

		const string tags0[]={"B", "E", "Je", "Ji"};
		float writebuffer[nzn-3][nyn-3][nxn-3][3];
		float tmpX, tmpY, tmpZ;

		for(int tagid=0;tagid<4;tagid++){
		 if (outputTag.find(tags0[tagid], 0) == string::npos) continue;

		 char   header[1024];
		 if (tags0[tagid].compare("B") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  tmpX = (float)EMf->getBxTot(ix+1, iy+1, iz+1);
						  tmpY = (float)EMf->getByTot(ix+1, iy+1, iz+1);
						  tmpZ = (float)EMf->getBzTot(ix+1, iy+1, iz+1);

						  writebuffer[iz][iy][ix][0] =  (fabs(tmpX) < 1E-16) ?0.0 : tmpX;
						  writebuffer[iz][iy][ix][1] =  (fabs(tmpY) < 1E-16) ?0.0 : tmpY;
						  writebuffer[iz][iy][ix][2] =  (fabs(tmpZ) < 1E-16) ?0.0 : tmpZ;
					  }

	      //Write VTK header
		  sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Magnetic Field from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d\n"
						   "VECTORS B float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (tags0[tagid].compare("E") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  writebuffer[iz][iy][ix][0] = (float)EMf->getEx(ix+1, iy+1, iz+1);
						  writebuffer[iz][iy][ix][1] = (float)EMf->getEy(ix+1, iy+1, iz+1);
						  writebuffer[iz][iy][ix][2] = (float)EMf->getEz(ix+1, iy+1, iz+1);
					  }

		  //Write VTK header
		   sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Electric Field from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS E float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (tags0[tagid].compare("Je") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  writebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 0);
						  writebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 0);
						  writebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 0);
					  }

		  //Write VTK header
		   sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Electron current from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS Je float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (tags0[tagid].compare("Ji") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  writebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 1);
						  writebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 1);
						  writebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 1);
					  }

		  //Write VTK header
		   sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Ion current from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS Ji float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);
		 }


		 if(EMf->isLittleEndian()){

			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  ByteSwap((unsigned char*) &writebuffer[iz][iy][ix][0],4);
						  ByteSwap((unsigned char*) &writebuffer[iz][iy][ix][1],4);
						  ByteSwap((unsigned char*) &writebuffer[iz][iy][ix][2],4);
					  }
		 }

		  int nelem = strlen(header);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_"<< tags0[tagid] << "_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

		  MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		  if (vct->getCartesian_rank()==0){
			  MPI_File_write(fh, header, nelem, MPI_BYTE, &status);
		  }

	      int err = MPI_File_set_view(fh, disp, EMf->getXYZeType(), EMf->getProcviewXYZ(), "native", MPI_INFO_NULL);
	      if(err){
	          dprintf("Error in MPI_File_set_view\n");

	          int error_code = status.MPI_ERROR;
	          if (error_code != MPI_SUCCESS) {
	              char error_string[100];
	              int length_of_error_string, error_class;

	              MPI_Error_class(error_code, &error_class);
	              MPI_Error_string(error_class, error_string, &length_of_error_string);
	              dprintf("Error %s\n", error_string);
	          }
	      }

	      err = MPI_File_write_all(fh, writebuffer[0][0][0],3*(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &status);
	      if(err){
		      int tcount=0;
		      MPI_Get_count(&status, MPI_DOUBLE, &tcount);
			  dprintf(" wrote %i",  tcount);
	          dprintf("Error in write1\n");
	          int error_code = status.MPI_ERROR;
	          if (error_code != MPI_SUCCESS) {
	              char error_string[100];
	              int length_of_error_string, error_class;

	              MPI_Error_class(error_code, &error_class);
	              MPI_Error_string(error_class, error_string, &length_of_error_string);
	              dprintf("Error %s\n", error_string);
	          }
	      }
	      MPI_File_close(&fh);
		}
	}

	if (outputTag.find("rho", 0) != string::npos){

		float writebufferRhoe[nzn-3][nyn-3][nxn-3];
		float writebufferRhoi[nzn-3][nyn-3][nxn-3];
		char   headerRhoe[1024];
		char   headerRhoi[1024];

		for(int iz=0;iz<nzn-3;iz++)
		  for(int iy=0;iy<nyn-3;iy++)
			  for(int ix= 0;ix<nxn-3;ix++){
				  writebufferRhoe[iz][iy][ix] = (float)EMf->getRHOns(ix+1, iy+1, iz+1, 0)*4*3.1415926535897;
				  writebufferRhoi[iz][iy][ix] = (float)EMf->getRHOns(ix+1, iy+1, iz+1, 1)*4*3.1415926535897;
			  }

		//Write VTK header
		sprintf(headerRhoe, "# vtk DataFile Version 2.0\n"
						   "Electron density from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "SCALARS rhoe float\n"
						   "LOOKUP_TABLE default\n",  dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		sprintf(headerRhoi, "# vtk DataFile Version 2.0\n"
						   "Ion density from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "SCALARS rhoi float\n"
						   "LOOKUP_TABLE default\n",  dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 if(EMf->isLittleEndian()){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  ByteSwap((unsigned char*) &writebufferRhoe[iz][iy][ix],4);
						  ByteSwap((unsigned char*) &writebufferRhoi[iz][iy][ix],4);
					  }
		 }

		  int nelem = strlen(headerRhoe);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_rhoe_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

		  MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		  if (vct->getCartesian_rank()==0){
			  MPI_File_write(fh, headerRhoe, nelem, MPI_BYTE, &status);
		  }

	      int err = MPI_File_set_view(fh, disp, MPI_FLOAT, EMf->getProcview(), "native", MPI_INFO_NULL);
	      if(err){
	          dprintf("Error in MPI_File_set_view\n");

	          int error_code = status.MPI_ERROR;
	          if (error_code != MPI_SUCCESS) {
	              char error_string[100];
	              int length_of_error_string, error_class;

	              MPI_Error_class(error_code, &error_class);
	              MPI_Error_string(error_class, error_string, &length_of_error_string);
	              dprintf("Error %s\n", error_string);
	          }
	      }

	      err = MPI_File_write_all(fh, writebufferRhoe[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &status);
	      if(err){
		      int tcount=0;
		      MPI_Get_count(&status, MPI_DOUBLE, &tcount);
			  dprintf(" wrote %i",  tcount);
	          dprintf("Error in write1\n");
	          int error_code = status.MPI_ERROR;
	          if (error_code != MPI_SUCCESS) {
	              char error_string[100];
	              int length_of_error_string, error_class;

	              MPI_Error_class(error_code, &error_class);
	              MPI_Error_string(error_class, error_string, &length_of_error_string);
	              dprintf("Error %s\n", error_string);
	          }
	      }
	      MPI_File_close(&fh);

	      nelem = strlen(headerRhoi);
	      disp  = nelem*charsize;

	      filename.str("");
	      filename << col->getSaveDirName() << "/" << col->getSimName() << "_rhoi_" << cycle << ".vtk";
	      MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

	      MPI_File_set_view(fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		  if (vct->getCartesian_rank()==0){
			  MPI_File_write(fh, headerRhoi, nelem, MPI_BYTE, &status);
		  }

	      err = MPI_File_set_view(fh, disp, MPI_FLOAT, EMf->getProcview(), "native", MPI_INFO_NULL);
		  if(err){
			  dprintf("Error in MPI_File_set_view\n");

			  int error_code = status.MPI_ERROR;
			  if (error_code != MPI_SUCCESS) {
				  char error_string[100];
				  int length_of_error_string, error_class;

				  MPI_Error_class(error_code, &error_class);
				  MPI_Error_string(error_class, error_string, &length_of_error_string);
				  dprintf("Error %s\n", error_string);
			  }
		  }

		  err = MPI_File_write_all(fh, writebufferRhoi[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &status);
		  if(err){
			  int tcount=0;
			  MPI_Get_count(&status, MPI_DOUBLE, &tcount);
			  dprintf(" wrote %i",  tcount);
			  dprintf("Error in write1\n");
			  int error_code = status.MPI_ERROR;
			  if (error_code != MPI_SUCCESS) {
				  char error_string[100];
				  int length_of_error_string, error_class;

				  MPI_Error_class(error_code, &error_class);
				  MPI_Error_string(error_class, error_string, &length_of_error_string);
				  dprintf("Error %s\n", error_string);
			  }
		  }
		  MPI_File_close(&fh);
	}
}

void WriteFieldsVTK(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, const string & outputTag ,int cycle,float**** fieldwritebuffer){

	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	MPI_File     fh;
	MPI_Status   status;
	const string fieldtags[]={"B", "E", "Je", "Ji","Je2", "Ji3"};
	const int    tagsize = size(fieldtags);
	const string outputtag = col->getFieldOutputTag();

	for(int tagid=0; tagid<tagsize; tagid++){

	 if (outputTag.find(fieldtags[tagid], 0) == string::npos) continue;

	 char   header[1024];
	 if (fieldtags[tagid].compare("B") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] =  (float)EMf->getBxTot(ix+1, iy+1, iz+1);
					  fieldwritebuffer[iz][iy][ix][1] =  (float)EMf->getByTot(ix+1, iy+1, iz+1);
					  fieldwritebuffer[iz][iy][ix][2] =  (float)EMf->getBzTot(ix+1, iy+1, iz+1);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
					   "Magnetic Field from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d\n"
					   "VECTORS B float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

	 }else if (fieldtags[tagid].compare("E") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getEx(ix+1, iy+1, iz+1);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getEy(ix+1, iy+1, iz+1);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getEz(ix+1, iy+1, iz+1);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
					   "Electric Field from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS E float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

	 }else if (fieldtags[tagid].compare("Je") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 0);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 0);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 0);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
					   "Electron current from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS Je float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

	 }else if (fieldtags[tagid].compare("Ji") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 1);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 1);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 1);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
					   "Ion current from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS Ji float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);
	 }else if (fieldtags[tagid].compare("Je2") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 2);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 2);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 2);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
				 "Electron2 current from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS Je float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

	 }else if (fieldtags[tagid].compare("Ji3") == 0){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 3);
					  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 3);
					  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 3);
				  }

		 //Write VTK header
		 sprintf(header, "# vtk DataFile Version 2.0\n"
				 "Ion3 current from iPIC3D\n"
					   "BINARY\n"
					   "DATASET STRUCTURED_POINTS\n"
					   "DIMENSIONS %d %d %d\n"
					   "ORIGIN 0 0 0\n"
					   "SPACING %f %f %f\n"
					   "POINT_DATA %d \n"
					   "VECTORS Ji float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);
	 }

	 if(EMf->isLittleEndian()){
		 for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++){
					  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][0],4);
					  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][1],4);
					  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][2],4);
				  }
	 }

	  int nelem = strlen(header);
	  int charsize=sizeof(char);
	  MPI_Offset disp = nelem*charsize;

	  ostringstream filename;
	  filename << col->getSaveDirName() << "/" << col->getSimName() << "_"<< fieldtags[tagid] << "_" << cycle << ".vtk";
	  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

	  if (vct->getCartesian_rank()==0){
		  MPI_File_write(fh, header, nelem, MPI_BYTE, &status);
	  }

      int error_code = MPI_File_set_view(fh, disp, EMf->getXYZeType(), EMf->getProcviewXYZ(), "native", MPI_INFO_NULL);
      if (error_code != MPI_SUCCESS) {
		char error_string[100];
		int length_of_error_string, error_class;

		MPI_Error_class(error_code, &error_class);
		MPI_Error_string(error_class, error_string, &length_of_error_string);
		dprintf("Error in MPI_File_set_view: %s\n", error_string);
	  }

      error_code = MPI_File_write_all(fh, fieldwritebuffer[0][0][0],(nxn-3)*(nyn-3)*(nzn-3),EMf->getXYZeType(), &status);
      if(error_code != MPI_SUCCESS){
	      int tcount=0;
	      MPI_Get_count(&status, EMf->getXYZeType(), &tcount);
          char error_string[100];
          int length_of_error_string, error_class;
          MPI_Error_class(error_code, &error_class);
          MPI_Error_string(error_class, error_string, &length_of_error_string);
          dprintf("Error in MPI_File_write_all: %s, wrote %d EMf->getXYZeType()\n", error_string,tcount);
      }
      MPI_File_close(&fh);
	}
}

void WriteMomentsVTK(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct, const string & outputTag ,int cycle, float*** momentswritebuffer){

	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	MPI_File     fh;
	MPI_Status   status;
	const string momentstags[]={"rho", "PXX", "PXY", "PXZ", "PYY", "PYZ", "PZZ"};
	const int    tagsize = size(momentstags);
	const string outputtag = col->getMomentsOutputTag();
	const int ns = col->getNs();

	for(int tagid=0; tagid<tagsize; tagid++){
		 if (outputtag.find(momentstags[tagid], 0) == string::npos) continue;

		 for(int si=0;si<ns;si++){
			 char  header[1024];
			 if (momentstags[tagid].compare("rho") == 0){
				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getRHOns(ix+1, iy+1, iz+1, si)*4*3.1415926535897;

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s%d density from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
					"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"rhoe":"rhoi");
		 }else if(momentstags[tagid].compare("PXX") == 0){
				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
				    for(int ix= 0;ix<nxn-3;ix++)
				      momentswritebuffer[iz][iy][ix] = (float)EMf->getpXXsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
						"%s%d pressure PXX from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
					"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXXe":"PXXi");
		}else if(momentstags[tagid].compare("PXY") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXYsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
					"%s%d pressure PXY from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXYe":"PXYi");
		}else if(momentstags[tagid].compare("PXZ") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXZsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
							   "%s%d pressure PXZ from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXZe":"PXZi");
		}else if(momentstags[tagid].compare("PYY") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpYYsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
							   "%s%d pressure PYY from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PYYe":"PYYi");
		}else if(momentstags[tagid].compare("PYZ") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpYZsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
							   "%s%d pressure PYZ from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PYZe":"PYZi");
		}else if(momentstags[tagid].compare("PZZ") == 0){

			for(int iz=0;iz<nzn-3;iz++)
			  for(int iy=0;iy<nyn-3;iy++)
				  for(int ix= 0;ix<nxn-3;ix++)
					  momentswritebuffer[iz][iy][ix] = (float)EMf->getpZZsn(ix+1, iy+1, iz+1, si);

			//Write VTK header
			sprintf(header, "# vtk DataFile Version 2.0\n"
							   "%s%d pressure PZZ from iPIC3D\n"
							   "BINARY\n"
							   "DATASET STRUCTURED_POINTS\n"
							   "DIMENSIONS %d %d %d\n"
							   "ORIGIN 0 0 0\n"
							   "SPACING %f %f %f\n"
							   "POINT_DATA %d \n"
							   "SCALARS %s float\n"
				"LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",si,dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PZZe":"PZZi");
		}

		 if(EMf->isLittleEndian()){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  ByteSwap((unsigned char*) &momentswritebuffer[iz][iy][ix],4);
					  }
		 }

		  int nelem = strlen(header);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_" << momentstags[tagid] << ((si%2==0)?"e":"i")<< si  << "_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

		  if (vct->getCartesian_rank()==0){
			  MPI_File_write(fh, header, nelem, MPI_BYTE, &status);
		  }

	      int error_code = MPI_File_set_view(fh, disp, MPI_FLOAT, EMf->getProcview(), "native", MPI_INFO_NULL);
	      if (error_code != MPI_SUCCESS) {
			char error_string[100];
			int length_of_error_string, error_class;
			MPI_Error_class(error_code, &error_class);
			MPI_Error_string(error_class, error_string, &length_of_error_string);
			dprintf("Error in MPI_File_set_view: %s\n", error_string);
		  }

	      error_code = MPI_File_write_all(fh, momentswritebuffer[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &status);
	      if(error_code != MPI_SUCCESS){
		      int tcount=0;
		      MPI_Get_count(&status, MPI_FLOAT, &tcount);
	          char error_string[100];
	          int length_of_error_string, error_class;
	          MPI_Error_class(error_code, &error_class);
	          MPI_Error_string(error_class, error_string, &length_of_error_string);
	          dprintf("Error in MPI_File_write_all: %s, wrote %d MPI_FLOAT\n", error_string,tcount);
	      }
	      MPI_File_close(&fh);
		 }//END OF SPECIES
	}//END OF TAGS
}


int WriteFieldsVTKNonblk(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct,int cycle,
		float**** fieldwritebuffer,MPI_Request requestArr[4],MPI_File fhArr[4]){

	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	const string fieldtags[]={"B", "E", "Je", "Ji"};
	const int fieldtagsize = size(fieldtags);
	const string fieldoutputtag = col->getFieldOutputTag();
	int counter=0,error_code;

	for(int tagid=0; tagid<fieldtagsize; tagid++){
		 if (fieldoutputtag.find(fieldtags[tagid], 0) == string::npos) continue;

		 fieldwritebuffer = &(fieldwritebuffer[counter]);
		 char  header[1024];
		 if (fieldtags[tagid].compare("B") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  fieldwritebuffer[iz][iy][ix][0] =  (float)EMf->getBxTot(ix+1, iy+1, iz+1);
						  fieldwritebuffer[iz][iy][ix][1] =  (float)EMf->getByTot(ix+1, iy+1, iz+1);
						  fieldwritebuffer[iz][iy][ix][2] =  (float)EMf->getBzTot(ix+1, iy+1, iz+1);
					  }

			 //Write VTK header
			 sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Magnetic Field from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d\n"
						   "VECTORS B float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (fieldtags[tagid].compare("E") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getEx(ix+1, iy+1, iz+1);
						  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getEy(ix+1, iy+1, iz+1);
						  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getEz(ix+1, iy+1, iz+1);
					  }

			 //Write VTK header
			 sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Electric Field from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS E float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (fieldtags[tagid].compare("Je") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 0);
						  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 0);
						  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 0);
					  }

			 //Write VTK header
			 sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Electron current from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS Je float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);

		 }else if (fieldtags[tagid].compare("Ji") == 0){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  fieldwritebuffer[iz][iy][ix][0] = (float)EMf->getJxs(ix+1, iy+1, iz+1, 1);
						  fieldwritebuffer[iz][iy][ix][1] = (float)EMf->getJys(ix+1, iy+1, iz+1, 1);
						  fieldwritebuffer[iz][iy][ix][2] = (float)EMf->getJzs(ix+1, iy+1, iz+1, 1);
					  }

			 //Write VTK header
			 sprintf(header, "# vtk DataFile Version 2.0\n"
						   "Ion current from iPIC3D\n"
						   "BINARY\n"
						   "DATASET STRUCTURED_POINTS\n"
						   "DIMENSIONS %d %d %d\n"
						   "ORIGIN 0 0 0\n"
						   "SPACING %f %f %f\n"
						   "POINT_DATA %d \n"
						   "VECTORS Ji float\n", dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints);
		 }

		 if(EMf->isLittleEndian()){
			 for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++){
						  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][0],4);
						  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][1],4);
						  ByteSwap((unsigned char*) &fieldwritebuffer[iz][iy][ix][2],4);
					  }
		 }

		  int nelem = strlen(header);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_"<< fieldtags[tagid] << "_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &(fhArr[counter]));

		  if (vct->getCartesian_rank()==0){
			  MPI_Status   status;
			  MPI_File_write(fhArr[counter], header, nelem, MPI_BYTE, &status);
		  }

		  error_code = MPI_File_set_view(fhArr[counter], disp, EMf->getXYZeType(), EMf->getProcviewXYZ(), "native", MPI_INFO_NULL);
	      if(error_code!= MPI_SUCCESS){
              char error_string[100];
              int length_of_error_string, error_class;
              MPI_Error_class(error_code, &error_class);
              MPI_Error_string(error_class, error_string, &length_of_error_string);
              dprintf("Error in MPI_File_set_view: %s\n", error_string);
	      }

	      error_code = MPI_File_write_all_begin(fhArr[counter],fieldwritebuffer[0][0][0],(nxn-3)*(nyn-3)*(nzn-3),EMf->getXYZeType());
	      //error_code = MPI_File_iwrite(fhArr[counter], fieldwritebuffer[0][0][0],(nxn-3)*(nyn-3)*(nzn-3),EMf->getXYZeType(), &(requestArr[counter]));
	      if(error_code!= MPI_SUCCESS){
              char error_string[100];
              int length_of_error_string, error_class;
              MPI_Error_class(error_code, &error_class);
              MPI_Error_string(error_class, error_string, &length_of_error_string);
              dprintf("Error in MPI_File_iwrite: %s", error_string);
	      }
	      counter ++;
		}
	return counter;
}


int  WriteMomentsVTKNonblk(Grid3DCU *grid, EMfields3D *EMf, CollectiveIO *col, VCtopology3D *vct,int cycle,
		float*** momentswritebuffer,MPI_Request requestArr[14],MPI_File fhArr[14]){

	//All VTK output at grid cells excluding ghost cells
	const int nxn  =grid->getNXN(),nyn  = grid->getNYN(),nzn =grid->getNZN();
	const int dimX =col->getNxc() ,dimY = col->getNyc(), dimZ=col->getNzc();
	const double spaceX = dimX>1 ?col->getLx()/(dimX-1) :col->getLx();
	const double spaceY = dimY>1 ?col->getLy()/(dimY-1) :col->getLy();
	const double spaceZ = dimZ>1 ?col->getLz()/(dimZ-1) :col->getLz();
	const int    nPoints = dimX*dimY*dimZ;
	const string momentstags[]={"rho", "PXX", "PXY", "PXZ", "PYY", "PYZ", "PZZ"};
	const int    tagsize = size(momentstags);
	const string outputtag = col->getMomentsOutputTag();
	int counter=0,err;

	for(int tagid=0; tagid<tagsize; tagid++){
		 if (outputtag.find(momentstags[tagid], 0) == string::npos) continue;

		 for(int si=0;si<=1;si++){
			 momentswritebuffer = &(momentswritebuffer[counter]);
			 char  header[1024];
			 if (momentstags[tagid].compare("rho") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getRHOns(ix+1, iy+1, iz+1, si)*4*3.1415926535897;

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s density from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"rhoe":"rhoi");
			 }else if(momentstags[tagid].compare("PXX") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXXsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PXX from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXXe":"PXXi");
			 }else if(momentstags[tagid].compare("PXY") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXYsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PXY from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXYe":"PXYi");
			}else if(momentstags[tagid].compare("PXZ") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpXZsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PXZ from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PXZe":"PXZi");
			}else if(momentstags[tagid].compare("PYY") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpYYsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PYY from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PYYe":"PYYi");
			}else if(momentstags[tagid].compare("PYZ") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpYZsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PYZ from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PYZe":"PYZi");
			}else if(momentstags[tagid].compare("PZZ") == 0){

				for(int iz=0;iz<nzn-3;iz++)
				  for(int iy=0;iy<nyn-3;iy++)
					  for(int ix= 0;ix<nxn-3;ix++)
						  momentswritebuffer[iz][iy][ix] = (float)EMf->getpZZsn(ix+1, iy+1, iz+1, si);

				//Write VTK header
				sprintf(header, "# vtk DataFile Version 2.0\n"
								   "%s pressure PZZ from iPIC3D\n"
								   "BINARY\n"
								   "DATASET STRUCTURED_POINTS\n"
								   "DIMENSIONS %d %d %d\n"
								   "ORIGIN 0 0 0\n"
								   "SPACING %f %f %f\n"
								   "POINT_DATA %d \n"
								   "SCALARS %s float\n"
								   "LOOKUP_TABLE default\n",(si%2==0)?"Electron":"Ion",dimX,dimY,dimZ, spaceX,spaceY,spaceZ, nPoints,(si%2==0)?"PZZe":"PZZi");
			}

			 if(EMf->isLittleEndian()){
				 for(int iz=0;iz<nzn-3;iz++)
					  for(int iy=0;iy<nyn-3;iy++)
						  for(int ix= 0;ix<nxn-3;ix++){
							  ByteSwap((unsigned char*) &momentswritebuffer[iz][iy][ix],4);
						  }
			 }

		  int nelem = strlen(header);
		  int charsize=sizeof(char);
		  MPI_Offset disp = nelem*charsize;

		  ostringstream filename;
		  filename << col->getSaveDirName() << "/" << col->getSimName() << "_" << momentstags[tagid] << ((si%2==0)?"e":"i") << "_" << cycle << ".vtk";
		  MPI_File_open(vct->getFieldComm(),filename.str().c_str(), MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fhArr[counter]);

		  if (vct->getCartesian_rank()==0){
			  MPI_Status   status;
			  MPI_File_write(fhArr[counter], header, nelem, MPI_BYTE, &status);
		  }

	      int error_code = MPI_File_set_view(fhArr[counter], disp, MPI_FLOAT, EMf->getProcview(), "native", MPI_INFO_NULL);
	      if (error_code != MPI_SUCCESS) {
			char error_string[100];
			int length_of_error_string, error_class;
			MPI_Error_class(error_code, &error_class);
			MPI_Error_string(error_class, error_string, &length_of_error_string);
			dprintf("Error in MPI_File_set_view: %s\n", error_string);
		  }

	      error_code = MPI_File_write_all_begin(fhArr[counter],momentswritebuffer[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT);
	      //error_code = MPI_File_iwrite(fhArr[counter], momentswritebuffer[0][0],(nxn-3)*(nyn-3)*(nzn-3),MPI_FLOAT, &(requestArr[counter]));
	      if(error_code != MPI_SUCCESS){
	          char error_string[100];
	          int length_of_error_string, error_class;
	          MPI_Error_class(error_code, &error_class);
	          MPI_Error_string(error_class, error_string, &length_of_error_string);
	          dprintf("Error in MPI_File_iwrite: %s \n", error_string);
	      }
	      counter ++;
		 }//END OF SPECIES
	}
	return counter;
}



void ByteSwap(unsigned char * b, int n)
{
   int i = 0;
   int j = n-1;
   while (i<j)
   {
      std::swap(b[i], b[j]);
      i++, j--;
   }
}
