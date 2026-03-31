<div align="center">
  <img src="fig/sensitive.png" width="50%"/>
  <p><b>Figure R1. Robustness to subtask annotation noise on LIBERO-Long.</b>
  Noise is injected by randomly replacing the subtask label and the object with probability (x%) (x ∈ {0, 2, 5, 10, 20}).</p>
</div>
  
--- 


<div>

<b>Table R1. Success rate (%) across VIMA-Bench task levels (L1–L4)</b>
Best results are <b><span style="color:#d62728;">bold (red)</span></b>, second-best are <i><span style="color:#1f77b4;">italic (blue)</span></i>.

<table>
<tr>
<th>Method</th>
<th>Sup.</th>
<th>Gen.</th>
<th>Align</th>
<th>Order</th>
<th>L1</th>
<th>L2</th>
<th>L3</th>
<th>L4</th>
</tr>

<tr>
<td>VIMA (Small)</td>
<td>None</td><td>--</td><td>--</td><td>--</td>
<td>72.4</td><td>73.8</td><td>71.1</td><td>35.0</td>
</tr>

<tr>
<td>VIMA (Large)</td>
<td>None</td><td>--</td><td>--</td><td>--</td>
<td>73.9</td><td>74.5</td><td>72.0</td><td><i><span style="color:#1f77b4;">59.0</span></i></td>
</tr>

<tr><td colspan="9" height="4"></td></tr>

<tr>
<td>Stacked VIMA</td>
<td>None</td><td>×</td><td>×</td><td>×</td>
<td>72.4</td><td>73.6</td><td>72.4</td><td>34.0</td>
</tr>

<tr>
<td>iSTAR (w/o Sup. & Align.)</td>
<td>None</td><td>×</td><td>×</td><td>✓</td>
<td>74.1</td><td>75.0</td><td>73.6</td><td>36.3</td>
</tr>

<tr>
<td>iSTAR (w/o Align. & Order)</td>
<td>Partial</td><td>✓</td><td>×</td><td>×</td>
<td>74.8</td><td>76.0</td><td>74.9</td><td>37.7</td>
</tr>

<tr>
<td>iSTAR (w/o Align.)</td>
<td>Partial</td><td>✓</td><td>×</td><td>✓</td>
<td>77.8</td><td>77.5</td><td>77.0</td><td>40.0</td>
</tr>

<tr>
<td>iSTAR (w/o Sup.)</td>
<td>Partial</td><td>×</td><td>✓</td><td>✓</td>
<td>76.7</td><td>77.5</td><td>76.4</td><td>41.3</td>
</tr>

<tr>
<td>iSTAR (w/o Order)</td>
<td>Full</td><td>✓</td><td>✓</td><td>×</td>
<td>76.4</td><td>77.6</td><td>76.3</td><td>42.7</td>
</tr>

<tr>
<td>iSTAR (Ours)</td>
<td>Full</td><td>✓</td><td>✓</td><td>✓</td>
<td><i><span style="color:#1f77b4;">78.6</span></i></td>
<td><i><span style="color:#1f77b4;">79.2</span></i></td>
<td><i><span style="color:#1f77b4;">78.3</span></i></td>
<td>44.3</td>
</tr>

<tr><td colspan="9" height="6"></td></tr>

<tr>
<td><b>iSTAR (Large Reasoner, Ours)</b></td>
<td>Full</td><td>✓</td><td>✓</td><td>✓</td>
<td><b><span style="color:#d62728;">78.8</span></b></td>
<td><b><span style="color:#d62728;">80.0</span></b></td>
<td><b><span style="color:#d62728;">81.1</span></b></td>
<td><b><span style="color:#d62728;">67.7</span></b></td>
</tr>

</table>
</div>

--- 

<div>

<b>Table R2. Success rate (SR, %) on LIBERO and CALVIN.</b>
Best results are <b><span style="color:#d62728;">bold (red)</span></b>, second-best are <i><span style="color:#1f77b4;">italic (blue)</span></i>.
<i>"w/o sup." denotes training without subtask-level supervision.</i>

<b>LIBERO</b>

<table>
<tr>
<th align="left">Method</th>
<th align="left">Params</th>
<th align="left">Spatial</th>
<th align="left">Object</th>
<th align="left">Goal</th>
<th align="left">Long</th>
<th align="left">Avg</th>
</tr>

<tr>
<td>OpenVLA-OFT</td>
<td>7B</td>
<td>96.2</td>
<td><i><span style="color:#1f77b4;">98.3</span></i></td>
<td><b><span style="color:#d62728;">96.2</span></b></td>
<td>90.7</td>
<td>95.3</td>
</tr>

<tr>
<td>iSTAR (OFT, w/o sup.)</td>
<td>8B</td>
<td><i><span style="color:#1f77b4;">96.6</span></i></td>
<td>98.2</td>
<td><b><span style="color:#d62728;">96.2</span></b></td>
<td><i><span style="color:#1f77b4;">91.8</span></i></td>
<td><i><span style="color:#1f77b4;">95.7</span></i></td>
</tr>

<tr>
<td>iSTAR (OFT, w/ sup.)</td>
<td>8B</td>
<td><b><span style="color:#d62728;">96.8</span></b></td>
<td><b><span style="color:#d62728;">98.4</span></b></td>
<td><i><span style="color:#1f77b4;">96.0</span></i></td>
<td><b><span style="color:#d62728;">92.2</span></b></td>
<td><b><span style="color:#d62728;">95.9</span></b></td>
</tr>

<tr><td colspan="7" height="1"></td></tr>

<tr>
<td>X-VLA</td>
<td>0.9B</td>
<td>98.1</td>
<td><i><span style="color:#1f77b4;">98.6</span></i></td>
<td>97.8</td>
<td>97.6</td>
<td>98.1</td>
</tr>

<tr>
<td>iSTAR (X-VLA, w/o sup.)</td>
<td>1B</td>
<td><i><span style="color:#1f77b4;">98.2</span></i></td>
<td><i><span style="color:#1f77b4;">98.6</span></i></td>
<td><b><span style="color:#d62728;">98.2</span></b></td>
<td><i><span style="color:#1f77b4;">98.0</span></i></td>
<td><i><span style="color:#1f77b4;">98.3</span></i></td>
</tr>

<tr>
<td>iSTAR (X-VLA, w/ sup.)</td>
<td>1B</td>
<td><b><span style="color:#d62728;">98.4</span></b></td>
<td><b><span style="color:#d62728;">99.0</span></b></td>
<td><i><span style="color:#1f77b4;">98.0</span></i></td>
<td><b><span style="color:#d62728;">98.6</span></b></td>
<td><b><span style="color:#d62728;">98.5</span></b></td>
</tr>

</table>


<b>CALVIN (ABC→D)</b>

<table>
<tr>
<th align="left">Method</th>
<th align="left">Params</th>
<th align="left">1</th>
<th align="left">2</th>
<th align="left">3</th>
<th align="left">4</th>
<th align="left">5</th>
<th align="left">Avg</th>
</tr>

<tr>
<td>X-VLA</td>
<td>0.9B</td>
<td>97.1</td>
<td>92.6</td>
<td>88.5</td>
<td>84.4</td>
<td>78.8</td>
<td>4.43</td>
</tr>

<tr>
<td>iSTAR (w/o sup.)</td>
<td>1B</td>
<td><b><span style="color:#d62728;">97.3</span></b></td>
<td><b><span style="color:#d62728;">93.8</span></b></td>
<td><b><span style="color:#d62728;">90.6</span></b></td>
<td><b><span style="color:#d62728;">87.9</span></b></td>
<td><b><span style="color:#d62728;">80.3</span></b></td>
<td><b><span style="color:#d62728;">4.50</span></b></td>
</tr>

</table>

</div>