import copy
import numpy as np
from scipy import ndimage as scnd
from scipy.interpolate import interp1d
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift


class AlignStack:

    def __init__(self, data, precision, trunc=False):  # Pass data as list
        self.Data = copy.deepcopy(data)  # Deepcopy avoids mutating input
        self.DataAligned = copy.deepcopy(data)
        self.precision = precision  # aligns to 1/precision pixels precise
        self.trunc=trunc

    def TwoDAlign(self):  # Translation alignment
        self.TList = ['']*len(self.Data)  # Save translation as x and y trace
        dset = 0  # counts the data sets in the Data list
        for d in self.Data:  # Loop through the Data list
            self.TList[dset] = np.zeros([d.shape[0], 2])
            D = d - scnd.gaussian_filter(d, 2)  # Take the high-F component
            print ('d D',d.shape,D.shape)
            template = D[0, :, :]  # Template is first image by default
            for i in range(1, d.shape[0]):  # Loop through stack and align
                # Align i'th image to template, save offset vector in TList
                self.TList[dset][i, :], error, difphase = register_translation(
                       template, D[i, :, :], upsample_factor=self.precision)
            self.TList[dset][2:, 0] = scnd.gaussian_filter(  # Smooth x y trace
                        self.TList[dset][:, 0],
                        2, mode='nearest', truncate=6)[2:]
            self.TList[dset][2:, 1] = scnd.gaussian_filter(
                        self.TList[dset][:, 1],
                        2, mode='nearest', truncate=6)[2:]

            for i in range(1, d.shape[0]):  # Loop through stack and register
                dPadded = np.pad(d[i, :, :], ((d.shape[1], d.shape[1]),
                                 (d.shape[2], d.shape[2])), 'reflect')
                ShiftedFFT = fourier_shift(
                        np.fft.fftn(dPadded), self.TList[dset][i, :])
                Shifted = np.absolute(np.fft.ifftn(ShiftedFFT))
                self.DataAligned[dset][i, :, :] = Shifted[
                        d.shape[1]:-d.shape[1], d.shape[2]:-d.shape[2]]
            # Cut-off parts that have missing pixels due to large shifts
            if self.trunc:
                BoundariesMin = np.round(np.max(self.TList[dset], axis=0))
                BoundariesMax = np.round(np.min(self.TList[dset], axis=0)) - 1

                self.DataAligned[dset] = self.DataAligned[dset][
                    :, int(BoundariesMin[0]):int(BoundariesMax[0]),
                    int(BoundariesMin[1]):int(BoundariesMax[1])]
            dset += 1

        self.Shifts = [np.zeros([d.shape[0], d.shape[1]]) for d in
                       self.DataAligned]  # Will contain calculated shifts
        self.ShiftsCumul = copy.deepcopy(self.Shifts)

    def RowAlign(self, interpol_c, MinimumShift, alfa):  # Align rows to templ.
        # Default: interpol_c = 50, MinimumShift = 0.2, alfa = 0.6
        # Uses interpolation (interpol_c subsamples) to find subpixel shifts
        # MinimumShift: treshold for which calc. shift is used for registration
        self.DataAligned2 = copy.deepcopy(self.DataAligned)  # Write to a copy
        self.RefEnergy = np.zeros(len(self.Data))  # Suitable as template image
        interpol_c=float(interpol_c)

        def NoiseTreshold(data, beta):  # Find noise treshold
            d = data.copy()

            if len(d.shape) == 2:  # If data is a matrix
                d = d.reshape(np.prod(d.shape))  # Make it into a vector

            P = np.percentile(abs(d), range(100))  # Calculate percentiles
            Pg = np.gradient(P)  # First derivative of percentiles
            Treshold = P[np.where(Pg > beta*np.mean(Pg))[0][0]]

            return(Treshold)

        def FindFourierShift(v1, v2):  # Finds shift between vector based on CC
            CC = np.fft.ifft(np.fft.fft(v2)*np.conj(np.fft.fft(v1)))
            Peak = np.where(CC == np.max(CC))[0]  # CC peak position = shift

            if Peak < (len(v1)/2):  # Positive shift
                s = Peak
            elif Peak > (len(v1)/2):  # Negative shift
                s = Peak - len(v1)

            return(s)

        def FindShifts(d, tem):  # Calculates all shifts for an energy stack
            print ("interp",interpol_c,1/interpol_c)
            dSum = np.sum(d, axis=0)  # Sum all energies of energy stack
            Treshold = NoiseTreshold(dSum, 1.2)  # Find noise treshold
            template = d[int(tem), :, :]  # Use first energy as template matrix
            xinter = np.arange(0, d.shape[2] - 1 + 2/interpol_c,
                               1/interpol_c)  # Interpolated x-axis
            xinter = xinter[xinter <= d.shape[2] - 1]
            f = interp1d(np.arange(0, d.shape[2]), template)  # Interpol. func.
            templateInterP = f(xinter)  # Interpolaterows of template matrix

            self.Shifts[dset] = np.zeros([d.shape[0], d.shape[1]])

            for i in [j for j in range(
                    0, d.shape[0]) if not j == tem]:  # Loop through energies
                f = interp1d(np.arange(0, d.shape[2]), d[i, :, :])
                DInterPMat = f(xinter)  # Interpol. rows of single-energy image
                for j in range(d.shape[1]):  # Loop through rows
                    if np.sum(dSum[j, :] > Treshold) > 2:  # Then signal in row
                        RowTInterP = templateInterP[j, :] - np.mean(
                                templateInterP[j, :])  # Template row
                        RowDInterP = DInterPMat[j, :] - np.mean(
                                DInterPMat[j, :])  # Row to be aligned
                        self.Shifts[dset][i, j] = FindFourierShift(
                                RowDInterP, RowTInterP)/interpol_c

        for dset in range(len(self.Data)):  # Loop data sets, exec. row alig.
            d = self.DataAligned[dset]
            print(dset)
            C = [0, 0]  # C is a cost measure
            It = 0  # Count the number of row alignment iterations

            while (It == 0) | (It == 1) | (  # Until converging C
                    (C[1] != 0) & (C[0] != C[1])):

                FindShifts(d, 0)  # Find Shifts matrix for d rel. to 1st energy

                TotalDev = np.sum(np.abs((self.Shifts[dset] - np.mean(
                        self.Shifts[dset], axis=0))), axis=1)
                self.RefEnergy[dset] = np.where(  # Is where shift is min
                        TotalDev == np.min(TotalDev))[0][0]

                self.Shifts[dset] = self.Shifts[dset] - self.Shifts[dset][
                        int(self.RefEnergy[dset]), :]  # Shifts rel. to RefEn.
                self.Shifts[dset][np.where(abs(  # If < MinimumShift, no shift
                        self.Shifts[dset]) < MinimumShift)] = 0

                C[0] = C[1]  # Shift costs to the left
                C[1] = np.sum(abs(self.Shifts[dset]))
                print(C)

                if (C[1] < C[0]) | (It == 0):
                    self.ShiftsCumul[dset] = self.ShiftsCumul[dset] +\
                        self.Shifts[dset]

                if (C[1] > C[0]) & (It != 0):
                    break

                for i in [j for j in range(0, d.shape[0]) if not
                          j == self.RefEnergy[dset]]:  # Loop through energ.
                    for j in [k for k in range(0, d.shape[1]) if self.Shifts[
                            dset][i, k] != 0]:  # Rows with non-zero shift
                        Row = d[i, j, :]  # Extract single row
                        RowPadded = np.pad(Row, len(Row), 'reflect')  # Pad row

                        # Shift in Fourier space:
                        RowShiftedFFT = fourier_shift(
                                np.fft.fft(RowPadded),
                                alfa*self.Shifts[dset][i, j])

                        RowShifted = np.absolute(np.fft.ifftn(RowShiftedFFT))
                        self.DataAligned2[dset][i, j, :] = RowShifted[
                                len(Row):-len(Row)]  # Remove pads

                self.DataAligned = copy.deepcopy(  # Allows successive runs
                        self.DataAligned2)
                d = self.DataAligned[dset]
                It += 1

    def Finalize(self):  # Cut off columns that miss pixels after alignment
        if not self.trunc:
            return
        for dset in range(len(self.DataAligned)):
            BoundariesMin = int(np.max(self.ShiftsCumul[dset]))
            BoundariesMax = int(np.min(self.ShiftsCumul[dset]) - 1)

            self.DataAligned[dset] = self.DataAligned[dset][
                        :, :, BoundariesMin:BoundariesMax]
