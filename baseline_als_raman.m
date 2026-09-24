function [Base, Corrected_Spectrum, info] = baseline_als_raman(Spectrum, lam, p, niter, tol)
% Asymmetric Least Squares baseline correction
% Based on Eilers & Boelens, 2005

    if nargin < 2 || isempty(lam),   lam = 1e5;   end
    if nargin < 3 || isempty(p),     p = 0.001;  end
    if nargin < 4 || isempty(niter), niter = 20; end
    if nargin < 5 || isempty(tol),   tol = 1e-6; end

    y = double(Spectrum(:));
    L = numel(y);

    lam = double(lam);
    p = double(p);
    niter = round(double(niter));
    tol = double(tol);

    if L < 3
        error('Spectrum must contain at least 3 points.');
    end

    if ~isscalar(lam) || ~isfinite(lam) || lam <= 0
        error('ALS lambda must be a positive scalar.');
    end

    if ~isscalar(p) || ~isfinite(p) || p <= 0 || p >= 1
        error('ALS p must be between 0 and 1.');
    end

    if ~isscalar(niter) || ~isfinite(niter) || niter < 1 || niter > 500
        error('ALS niter must be between 1 and 500.');
    end

    if ~isscalar(tol) || ~isfinite(tol) || tol <= 0
        error('ALS tol must be positive.');
    end

    if any(~isfinite(y))
        x = (1:L)';
        good = isfinite(y);
        if nnz(good) < 3
            error('Spectrum has fewer than 3 finite points.');
        end
        y(~good) = interp1(x(good), y(good), x(~good), 'linear', 'extrap');
    end

    D = diff(speye(L), 2, 1);
    H = lam * (D' * D);

    w = ones(L,1);
    Base = zeros(L,1);

    info.rel_change = nan(niter,1);

    for k = 1:niter
        Base_old = Base;

        W = spdiags(w, 0, L, L);
        Base = (W + H) \ (w .* y);

        if k > 1
            rel_change = norm(Base - Base_old) / max(norm(Base_old), eps);
            info.rel_change(k) = rel_change;

            if rel_change < tol
                break
            end
        end

        w = (1 - p) * ones(L,1);
        w(y > Base) = p;
    end

    Corrected_Spectrum = y - Base;

    info.iterations = k;
    info.rel_change = info.rel_change(1:k);
    info.lam = lam;
    info.p = p;
end