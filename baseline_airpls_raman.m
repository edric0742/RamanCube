function [Base, Corrected_Spectrum, info] = baseline_airpls_raman(Spectrum, lam, niter, ratio)
% airPLS baseline correction for Raman spectra
% Adaptive Iteratively Reweighted Penalized Least Squares

    if nargin < 2 || isempty(lam),   lam = 1e5;     end
    if nargin < 3 || isempty(niter), niter = 50;    end
    if nargin < 4 || isempty(ratio), ratio = 1e-3;  end

    y = Spectrum(:);
    L = length(y);

    if L < 3
        error('Spectrum must contain at least 3 points.');
    end

    D = diff(speye(L), 2);
    P = lam * (D' * D);

    w = ones(L,1);
    total_abs_y = sum(abs(y));

    info.negative_residual_sum = zeros(niter,1);

    for k = 1:niter
        W = spdiags(w, 0, L, L);
        Base = (W + P) \ (w .* y);

        d = y - Base;
        neg_idx = d < 0;
        dssn = sum(abs(d(neg_idx)));

        info.negative_residual_sum(k) = dssn;

        if dssn < ratio * total_abs_y
            break;
        end

        w(d >= 0) = 0;
        w(neg_idx) = exp(k * abs(d(neg_idx)) / max(dssn, eps));

        w(1) = max(w);
        w(end) = max(w);
    end

    Corrected_Spectrum = y - Base;

    info.iterations = k;
    info.negative_residual_sum = info.negative_residual_sum(1:k);
    info.lam = lam;
    info.ratio = ratio;
end