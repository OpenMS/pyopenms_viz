For testing pyopenms_viz uses a snapshot-based tests with a techstack including pytest and syrupy.

Custom Syrupy snapshot extensions for Bokeh, Matplotlib and Plotly are used. 
Testing can be done by running 
```bash 
# from the main directory
pytest . 
```
It is important that the exact versions in the `requirements.txt` file are used for snapshots as visual snapshots are quite finiky. 

Sometimes with version updates snapshots need to be updated. The best way to do this is by moving the current snapshots to an alternative folder, generating new snapshots and then visually comparing these snapshots. For example:

```bash
# from main directory 
mv tests/__snapshots__ tests/__snapshots_OLD 
pytest --snapshot-update
```

There will then be two folders of snapshots which can be visually compared. For visual comparison it is recommended to use the notebooks in the `nbs/` folder.

Happy testing!
