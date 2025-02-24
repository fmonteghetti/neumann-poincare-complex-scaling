/*********************************************************************
 * Unstructured mesh of an ellipse.
 *  - No symmetry is enforced.
 *  - Element size controlled by curvature.
 */

SetFactory("OpenCASCADE");
    // ---- Input parameters

    // Ellipse gamma-m semi-axes
If (!Exists(a_m)) a_m=2.5; EndIf
If (!Exists(b_m)) b_m=1; EndIf
    // Ellipse gamma-d semi-axes
If (!Exists(a_d)) a_d=1.3*a_m; EndIf
If (!Exists(b_d)) b_d=Sqrt(a_d^2-Abs(a_m^2-b_m^2)); EndIf
    // Number of element on gamma_m
If (!Exists(N_m)) N_m=20; EndIf
    
    // ---- Ellipses
l_d=newl; Ellipse(l_d) = {0, 0, 0, a_d, b_d, 0, 2*Pi};
l_m=newl; Ellipse(l_m) = {0, 0, 0, a_m, b_m, 0, 2*Pi};

    // Internal ellipse
ll_m = newll; Curve Loop(ll_m) = {l_m};
s_m = news; Surface(s_m) = {ll_m};
ll_d = newll; Curve Loop(ll_d) = {l_d};
s_d = news; Plane Surface(s_d) = {ll_d, ll_m};

    // -- Define Physical Entitites
Physical Curve("gamma-m") = {l_m};
Physical Curve("gamma-d") = {l_d};
Physical Surface("omega-m") = {s_m};
Physical Surface("omega-d") = {s_d};

    // -- Meshing
Mesh.MeshSizeFromCurvature = N_m;