function [Base, Corrected_Spectrum, info] = baseline_arpls_raman(Spectrum, lam, ratio, niter)
% arPLS baseline correction for Raman spectra
% Adaptive reweighted Penalized Least Squares

    if nargin < 2 || isempty(lam),   lam = 1e5;    end
    if nargin < 3 || isempty(ratio), ratio = 1e-6;  end
    if nargin < 4 || isempty(niter), niter = 50;    end

    y = Spectrum(:);
    L = length(y);

    if L < 3
        error('Spectrum must contain at least 3 points.');
    end

    D = diff(speye(L), 2);
    P = lam * (D' * D);

    w = ones(L,1);
    info.rel_change = zeros(niter,1);

    for k = 1:niter
        W = spdiags(w, 0, L, L);
        Base = (W + P) \ (w .* y);

        d = y - Base;
        dn = d(d < 0);

        if isempty(dn)
            break;
        end

        m = mean(dn);
        s = std(dn);

        if s < eps
            break;
        end

        w_new = 1 ./ (1 + exp(2 * (d - (2*s - m)) / s));

        rel_change = norm(w_new - w) / max(norm(w), eps);
        info.rel_change(k) = rel_change;

        w = w_new;

        if rel_change < ratio
            break;
        end
    end

    Corrected_Spectrum = y - Base;

    info.iterations = k;
    info.rel_change = info.rel_change(1:k);
    info.lam = lam;
    info.ratio = ratio;
end