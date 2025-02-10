# superNISP: Euclid-NISP to JWST-NIRCAM super resolution

Aim: Weak lensing is a science driver of Euclid. Right now Euclid uses VIS (optical) imaging for weak lensing due to its higher resolution compared to NISP (NIR). However being able to do the accurate shape measurements in the IR, enables a larger sample size for weak lensing and hence tighter cosmology constraints (e.g., Lee et al. 2018, Finner et al. 2023). 
Through this project, we aim to train a deep learning network to do super resolution on Euclid NISP imaging by learning from a subset that have JWST imaging at the same wavelengths.


Data:
High-resolution imaging (JWST NIRCAM)
Low-resolution imaging (Euclid NISP)
Catalog for selection:
Cosmos field seems to be a great choice covered by both JWST through cosmos-web program, and by Euclid. But I will double check and make sure if there’s a better choice. Such as the Euclid deep field north.
Cosmos2020 is the name of the catalog in the cosmos field, 

Method:

We need a super resolution network and can come up with the best option, we can start with a conditional GAN, a conditional DDPM, or be very ambitious and do foundation models, examples below:
cGAN: https://github.com/xoubish/disks
DDPM: https://github.com/Smith42/astroddpm
Multi-modal: https://github.com/PolymathicAI/AstroCLIP
