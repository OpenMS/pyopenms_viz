from __future__ import annotations
import logging as logging
from rdkit.Chem import rdchem
import rdkit.Chem.rdchem
from rdkit.Chem import rdinchi
from rdkit import RDLogger
import rdkit.RDLogger
import typing as typing
__all__: list = ['MolToInchiAndAuxInfo', 'MolToInchi', 'MolBlockToInchiAndAuxInfo', 'MolBlockToInchi', 'MolFromInchi', 'InchiReadWriteError', 'InchiToInchiKey', 'MolToInchiKey', 'GetInchiVersion', 'INCHI_AVAILABLE']
class InchiReadWriteError(Exception):
    pass
def InchiToInchiKey(inchi: str) -> typing.Optional[str]:
    """
    Return the InChI key for the given InChI string. Return None on error
    """
def MolBlockToInchi(molblock: str, options: str = '', logLevel: typing.Optional[int] = None, treatWarningAsError: bool = False) -> str:
    """
    Returns the standard InChI string for a mol block
    
        Keyword arguments:
        logLevel -- the log level used for logging logs and messages from InChI
        API. set to None to diable the logging completely
        treatWarningAsError -- set to True to raise an exception in case of a
        molecule that generates warning in calling InChI API. The resultant InChI
        string and AuxInfo string as well as the error message are encoded in the
        exception.
    
        Returns:
        the standard InChI string returned by InChI API for the input molecule
        
    """
def MolBlockToInchiAndAuxInfo(molblock: str, options: str = '', logLevel: typing.Optional[int] = None, treatWarningAsError: bool = False) -> typing.Tuple[str, str]:
    """
    Returns the standard InChI string and InChI auxInfo for a mol block
    
        Keyword arguments:
        logLevel -- the log level used for logging logs and messages from InChI
        API. set to None to diable the logging completely
        treatWarningAsError -- set to True to raise an exception in case of a
        molecule that generates warning in calling InChI API. The resultant InChI
        string and AuxInfo string as well as the error message are encoded in the
        exception.
    
        Returns:
        a tuple of the standard InChI string and the auxInfo string returned by
        InChI API, in that order, for the input molecule
        
    """
def MolFromInchi(inchi: str, sanitize: bool = True, removeHs: bool = True, logLevel: typing.Optional[int] = None, treatWarningAsError: bool = False) -> typing.Optional[rdkit.Chem.rdchem.Mol]:
    """
    Construct a molecule from a InChI string
    
        Keyword arguments:
        sanitize -- set to True to enable sanitization of the molecule. Default is
        True
        removeHs -- set to True to remove Hydrogens from a molecule. This only
        makes sense when sanitization is enabled
        logLevel -- the log level used for logging logs and messages from InChI
        API. set to None to diable the logging completely
        treatWarningAsError -- set to True to raise an exception in case of a
        molecule that generates warning in calling InChI API. The resultant
        molecule  and error message are part of the excpetion
    
        Returns:
        a rdkit.Chem.rdchem.Mol instance
        
    """
def MolToInchi(mol: rdkit.Chem.rdchem.Mol, options: str = '', logLevel: typing.Optional[int] = None, treatWarningAsError: bool = False) -> str:
    """
    Returns the standard InChI string for a molecule
    
        Keyword arguments:
        logLevel -- the log level used for logging logs and messages from InChI
        API. set to None to diable the logging completely
        treatWarningAsError -- set to True to raise an exception in case of a
        molecule that generates warning in calling InChI API. The resultant InChI
        string and AuxInfo string as well as the error message are encoded in the
        exception.
    
        Returns:
        the standard InChI string returned by InChI API for the input molecule
        
    """
def MolToInchiAndAuxInfo(mol: rdkit.Chem.rdchem.Mol, options: str = '', logLevel: typing.Optional[int] = None, treatWarningAsError: bool = False) -> typing.Tuple[str, str]:
    """
    Returns the standard InChI string and InChI auxInfo for a molecule
    
        Keyword arguments:
        logLevel -- the log level used for logging logs and messages from InChI
        API. set to None to diable the logging completely
        treatWarningAsError -- set to True to raise an exception in case of a
        molecule that generates warning in calling InChI API. The resultant InChI
        string and AuxInfo string as well as the error message are encoded in the
        exception.
    
        Returns:
        a tuple of the standard InChI string and the auxInfo string returned by
        InChI API, in that order, for the input molecule
        
    """
def MolToInchiKey(mol: rdkit.Chem.rdchem.Mol, options: str = '') -> typing.Optional[str]:
    """
    Returns the standard InChI key for a molecule
    
        Returns:
        the standard InChI key returned by InChI API for the input molecule
        
    """
INCHI_AVAILABLE: bool = True
logLevelToLogFunctionLookup: dict = {20: rdkit.RDLogger.logger.info, 10: rdkit.RDLogger.logger.debug, 30: rdkit.RDLogger.logger.warning, 50: rdkit.RDLogger.logger.critical, 40: rdkit.RDLogger.logger.error}
logger: rdkit.RDLogger.logger  # value = <rdkit.RDLogger.logger object>
