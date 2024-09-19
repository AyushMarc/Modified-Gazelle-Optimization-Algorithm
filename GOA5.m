function [Top_gazelle_fit, Top_gazelle_pos, Convergence_curve] = GOA5(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

    Top_gazelle_pos = zeros(1, dim);
    Top_gazelle_fit = inf;
    Convergence_curve = zeros(1, Max_iter);
    stepsize = zeros(SearchAgents_no, dim);
    fitness = inf(SearchAgents_no, 1);

    gazelle = initialization(SearchAgents_no, dim, ub, lb);

    Xmin = repmat(ones(1, dim) .* lb, SearchAgents_no, 1);
    Xmax = repmat(ones(1, dim) .* ub, SearchAgents_no, 1);

    Iter = 0;
    PSRs = 0.34;
    S = 0.88;
    s = rand();

    while Iter < Max_iter
        % Perform exploration and exploitation multiple times in a row
        for k = 1:4
            %------------------- Evaluating top gazelle -----------------
            for i = 1:size(gazelle, 1)
                Flag4ub = gazelle(i, :) > ub;
                Flag4lb = gazelle(i, :) < lb;
                gazelle(i, :) = (gazelle(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
                fitness(i, 1) = fobj(gazelle(i, :));
                if fitness(i, 1) < Top_gazelle_fit
                    Top_gazelle_fit = fitness(i, 1);
                    Top_gazelle_pos = gazelle(i, :);
                end
            end

            %------------------- Keeping track of fitness values-------------------
            if Iter == 0
                fit_old = fitness;
                Prey_old = gazelle;
            end

            Inx = (fit_old < fitness);
            Indx = repmat(Inx, 1, dim);
            gazelle = Indx .* Prey_old + ~Indx .* gazelle;
            fitness = Inx .* fit_old + ~Inx .* fitness;

            fit_old = fitness;
            Prey_old = gazelle;

            %------------------------------------------------------------

            Elite = repmat(Top_gazelle_pos, SearchAgents_no, 1);  %(Eq. 3)
            CF = (1 - Iter / Max_iter) ^ (2 * Iter / Max_iter);

            RL = levy(SearchAgents_no, dim, 1.5);   % Levy random number vector
            IDS_depth = 5;  % Maximum depth for IDS

            for i = 1:size(gazelle, 1)
                for j = 1:size(gazelle, 2)
                    R = rand();
                    r = rand();
                    if mod(Iter, 2) == 0
                        mu = -1;
                    else
                        mu = 1;
                    end
                    %------------------ Exploitation -------------------
                    if r > 0.5
                        stepsize(i, j) = RL(i, j) * (Elite(i, j) - RL(i, j) * gazelle(i, j));
                        gazelle(i, j) = gazelle(i, j) + s * R * stepsize(i, j);

                    %--------------- Exploration using IDS----------------
                    else
                        found_better = false;
                        for depth = 1:IDS_depth
                            new_pos = gazelle(i, :);
                            new_pos(j) = new_pos(j) + RL(i, j) * mu * depth;

                            % Check if the new position is within bounds
                            new_pos = max(min(new_pos, ub), lb);

                            % Evaluate fitness of the new position
                            new_fitness = fobj(new_pos);

                            % If the new position is better, update
                            if new_fitness < fitness(i, 1)
                                gazelle(i, :) = new_pos;
                                fitness(i, 1) = new_fitness;
                                found_better = true;
                                break;  % Exit loop if a better solution is found
                            end
                        end

                        % If no better solution was found within depth, move randomly
                        if ~found_better
                            stepsize(i, j) = RL(i, j) * (Elite(i, j) - RL(i, j) * gazelle(i, j));
                            gazelle(i, j) = gazelle(i, j) + S * mu * R * stepsize(i, j);
                        end
                    end
                end
            end

            %------------------ Updating top gazelle ------------------
            for i = 1:size(gazelle, 1)
                Flag4ub = gazelle(i, :) > ub;
                Flag4lb = gazelle(i, :) < lb;
                gazelle(i, :) = (gazelle(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
                fitness(i, 1) = fobj(gazelle(i, :));
                if fitness(i, 1) < Top_gazelle_fit
                    Top_gazelle_fit = fitness(i, 1);
                    Top_gazelle_pos = gazelle(i, :);
                end
            end

            %---------------------- Updating history of fitness values ----------------
            if Iter == 0
                fit_old = fitness;
                Prey_old = gazelle;
            end

            Inx = (fit_old < fitness);
            Indx = repmat(Inx, 1, dim);
            gazelle = Indx .* Prey_old + ~Indx .* gazelle;
            fitness = Inx .* fit_old + ~Inx .* fitness;

            fit_old = fitness;
            Prey_old = gazelle;

            %---------- Applying PSRs -----------
            if rand() < PSRs
                U = rand(SearchAgents_no, dim) < PSRs;
                gazelle = gazelle + CF * ((Xmin + rand(SearchAgents_no, dim) .* (Xmax - Xmin)) .* U);
            else
                r = rand();  Rs = size(gazelle, 1);
                stepsize = (PSRs * (1 - r) + r) * (gazelle(randperm(Rs), :) - gazelle(randperm(Rs), :));
                gazelle = gazelle + stepsize;
            end
        end

        Iter = Iter + 1;
        Convergence_curve(Iter) = Top_gazelle_fit;
    end

end

function [RL] = levy(SearchAgents_no, dim, scale)
    RL = scale * randn(SearchAgents_no, dim);
end

function [Population] = initialization(PopSize, dim, ub, lb)
    Population = repmat(lb, PopSize, 1) + rand(PopSize, dim) .* (repmat(ub - lb, PopSize, 1));
end