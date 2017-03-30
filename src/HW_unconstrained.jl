

module HW_unconstrained

	using Distributions, Optim, DataFrames
	using Gadfly
	using Base.Test
	#using ForwardDiff
	#using Debug

	"""
    `input(prompt::AbstractString="")`

    Read a string from STDIN. The trailing newline is stripped.

    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
  """
  function input(prompt::AbstractString="")
      print(prompt)
      return chomp(readline())
	end

  export maximize_like_grad, runAll, makeData


	# methods/functions
	# -----------------

	# DATA CREATOR
	function makeData(n=1000)

		srand(1234)
		# Parameters
		beta = [1.0; 1.5; -0.5]
		mu = [0.0;0.0;0.0]
		sigma = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

		# Drawing the data X
		dx = MvNormal(mu, sigma)
		X = rand(dx, n)

		# Computing the associated response y
		d = Normal(0,1)
		proba = cdf(d, transpose(X)*beta)
		y = zeros(n)
		for i = 1:n
			dy = Binomial(1,proba[i])
			y[i] = rand(dy)
		end

		return Dict("coeff"=> beta,"nobs"=> n, "data"=> X, "response" => y, "distrib" => d)

	end

	# LOG LIKELIHOOD FUNCTION AT X
	function loglik(betas::Vector,d::Dict)
		loglik = 0
		loglik = dot( d["response"], log(cdf(d["distrib"], transpose(d["data"])*betas)) ) + dot( (1-d["response"]), log( 1 - cdf(d["distrib"],transpose(d["data"])*betas) ) )
		return loglik
	end

	# Gradient of the LOG-likelihood at x
	function grad!(betas::Vector,storage::Vector,d)

		y = d["response"] # Dimension = (N,1)
		X = d["data"] # Dimension = (3,N)
		distrib = d["distrib"]

		phi = cdf(distrib, transpose(X)*betas) # Dimension = (N,1)
		phi_deriv = pdf(distrib, transpose(X)*betas) # Dimension = (N,1)

		storage[1] = -(transpose((y./phi).*phi_deriv)*X[1,:] - transpose(((1-y)./(1-phi).*phi_deriv))*X[1,:])[1]
		storage[2] = -(transpose((y./phi).*phi_deriv)*X[2,:] - transpose(((1-y)./(1-phi).*phi_deriv))*X[2,:])[1]
		storage[3] = -(transpose((y./phi).*phi_deriv)*X[3,:] - transpose(((1-y)./(1-phi).*phi_deriv))*X[3,:])[1]
		storage = [storage[1]; storage[2]; storage[3]]

	end

	# Hessian of the LOG-likelihood at x
	function hessian!(betas::Vector,storage::Matrix,d::Dict)

		y = d["response"] # Dimension = (N,1)
		X = d["data"] # Dimension = (3,N)
		distrib = d["distrib"]

		phi = cdf(distrib, transpose(X)*betas) # Dimension = (N,1)
		phi_deriv = pdf(distrib, transpose(X)*betas) # Dimension = (N,1)
		phi_deriv_deriv = - (transpose(X)*betas).*phi_deriv # Dimension = (N,1)

		term1 = (transpose(y./phi) * (phi_deriv_deriv - (phi_deriv.*phi_deriv)./phi))
		term2 = (transpose((1-y)./(1-phi)) * (phi_deriv_deriv - (phi_deriv.*phi_deriv)./(1-phi)))
		storage = (term1 - term2).* (X * transpose(X))
		#storage = ( (transpose(y./phi) * (phi_deriv_deriv - phi_deriv.*phi_deriv./phi)) - (transpose((1-y)./(1-phi)) * (phi_deriv_deriv - phi_deriv.*phi_deriv./(1-phi))) ) .* X*transpose(X)
		storage = - storage

	end


	function inv_observedInfo(betas::Vector,d)
	end

	function se(betas::Vector,d::Dict)
	end

	function maximize_like(x0=[0.8,1.0,-0.1],meth=:"NelderMead")

		srand(1234)
		d = makeData()
		f = (x) -> -loglik(x,d) # Changing the objective function since Optim returns the min and not the max of a function
		if meth == "NelderMead"
			min = optimize(f,x0, NelderMead())
			return min
		end

	end

	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=:"GradientDescent")

		srand(1234)
		d = makeData()
		f = (b) -> -loglik(b,d) # Changing the objective function since Optim returns the min and not the max of a function
		g! = (b, storage) -> grad!(b,storage,d) # already defined in minus

		if meth == "GradientDescent"
			min = optimize(f, g!, x0, GradientDescent())
			return min
		end

		if meth == "BFGS"
			min = optimize(f, g!, x0, BFGS())
			return min
		end

	end

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=:"LBFGS")

		srand(1234)
		d = makeData()
		f = (b) -> -loglik(b,d) # Changing the objective function since Optim returns the min and not the max of a function
		g! = (b, storage) -> grad!(b, storage, d) # already defined in minus
		h! = (b, storage) -> hessian!(b, storage, d) # already defined in minus

		if meth == "Newton"
			min = optimize(f, g!, h!, x0, Newton())
			return min
		end

		if meth == "LBFGS"
			min = optimize(f, g!, h!, x0, LBFGS())
			return min
		end

	end

	#function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=:bfgs)
	#end


	# VISUAL DIAGNOSTICS
	# ------------------

	# function that plots the likelihood
	function plotLike(lb=-3,ub=3)

		srand(1234)
		d = makeData()
		beta1_true = d["coeff"][1]
		beta2_true = d["coeff"][2]
		beta3_true = d["coeff"][3]
		f1 = (beta1) -> loglik([beta1; beta2_true; beta3_true], d)
		f2 = (beta2) -> loglik([beta1_true; beta2; beta3_true], d)
		f3 = (beta3) -> loglik([beta1_true; beta2_true; beta3], d)

		beta = linspace(lb,ub,d["nobs"])
		dataset = DataFrame(beta_val = beta, loglik1 = f1.(beta), loglik2 = f2.(beta), loglik3 = f3.(beta))
		plot1 = Gadfly.plot(dataset, x = "beta_val", y = "loglik1", xintercept = [beta1_true], Geom.point, Geom.vline(color="orange", size=1.5mm), Guide.ylabel("Log-likelihood"), Guide.xlabel("Beta 1"), Theme(default_point_size = 1pt, default_color = colorant"blue",highlight_width = 0.1pt))
		plot2 = Gadfly.plot(dataset, x = "beta_val", y = "loglik2", xintercept = [beta2_true], Geom.point, Geom.vline(color="orange", size=1.5mm), Guide.ylabel("Log-likelihood"), Guide.xlabel("Beta 2"), Theme(default_point_size = 1pt, default_color = colorant"blue",highlight_width = 0.1pt))
		plot3 = Gadfly.plot(dataset, x = "beta_val", y = "loglik3", xintercept = [beta3_true], Geom.point, Geom.vline(color="orange", size=1.5mm), Guide.ylabel("Log-likelihood"), Guide.xlabel("Beta 3"), Theme(default_point_size = 1pt, default_color = colorant"blue",highlight_width = 0.1pt))
		display(plot1)
		display(plot2)
		display(plot3)
		hstack(plot1,plot2,plot3)

	end

	function plotGrad(lb=-2, ub=2)

		d = makeData()
		storage = zeros(3)

		beta1_true = d["coeff"][1]; beta2_true = d["coeff"][2]; beta3_true = d["coeff"][3]
		f1 = (beta1) -> grad!([beta1; beta2_true; beta3_true], storage, d)[1]
		f2 = (beta2) -> grad!([beta1_true; beta2; beta3_true], storage, d)[2]
		f3 = (beta3) -> grad!([beta1_true; beta2_true; beta3], storage, d)[3]

		beta = linspace(lb,ub,d["nobs"])
		dataset = DataFrame(beta_val = beta, grad1 = f1.(beta), grad2 = f2.(beta), grad3 = f3.(beta))
		plot1 = Gadfly.plot(dataset, x = "beta_val", y = "grad1", xintercept = [beta1_true], Geom.point, Geom.vline(color="orange", size=1.5mm), Guide.ylabel("Gradient"), Guide.xlabel("Beta 1"), Guide.Title("Gradient & True Beta"), Theme(default_point_size = 1pt, default_color = colorant"blue",highlight_width = 0.1pt))
		plot2 = Gadfly.plot(dataset, x = "beta_val", y = "grad2", xintercept = [beta2_true], Geom.point, Geom.vline(color="orange", size=1.5mm), Guide.ylabel("Gradien"), Guide.xlabel("Beta 2"), Guide.Title("Gradient & True Beta"), Theme(default_point_size = 1pt, default_color = colorant"blue",highlight_width = 0.1pt))
		plot3 = Gadfly.plot(dataset, x = "beta_val", y = "grad3", xintercept = [beta3_true], Geom.point, Geom.vline(color="orange", size=1.5mm), Guide.ylabel("Gradient"), Guide.xlabel("Beta 3"), Guide.Title("Gradient & True Beta"), Theme(default_point_size = 1pt, default_color = colorant"blue",highlight_width = 0.1pt))
		display(plot1)
		display(plot2)
		display(plot3)
		hstack(plot1,plot2,plot3)

	end


	function runAll()

		srand(1234)
		data = makeData()

		plotLike()
		plotGrad()

		#println(grad!([0.8;1.0;-0.1], zeros(3), data))
		#println(hessian!([0.8;1.0;-0.1], zeros(3,3), data))

		m1 = maximize_like().minimizer
		m2 = maximize_like_grad().minimizer
		m3 = maximize_like_grad_hess().minimizer
		#m4 = maximize_like_grad_se()
		println("*****************************************************************")
		println("results are:")
		println("--------------")
		println("Raw Method : ")
		println("maximize_like: $m1 ")
		println("--------------")
		println("Gradient Method : ")
		println("maximize_like_grad: $m2")
		println("--------------")
		println("Gradient + Hessian Method : ")
		println("maximize_like_grad_hess: $m3 ")
		#println("maximize_like_grad_se: $m4")
		println("*****************************************************************")
		println("running tests:")
		include("test/runtests.jl")
		#println("")
		#ok = input("enter y to close this session.")
		#if ok == "y"
			#quit()
		#end

	end


end
