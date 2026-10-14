* Fixed USB routing cable body selection for Newton's free-root articulation layout,
  restoring the anchored first segment and excluding unrelated bodies from cable mass
  and contact bookkeeping. Updated contact linkage authoring to the current MuJoCo
  equality-type API.
* Restricted cable stiffness randomization to rod joints, excluding the free root
  joint degrees of freedom.
