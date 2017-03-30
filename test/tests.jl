
using HW_unconstrained
using Base.Test
using Distributions
using Optim

println("Running the tests")
@testset "basics" begin

	@testset "Test Data Construction" begin

		srand(1234)
		d = makeData()
		# Testing the dimensions of the data
		@test length(d["data"]) == 3*d["nobs"]
		@test length(d["response"]) == d["nobs"]
		# Testing that the reponses simulated are either 0 and 1 (indeed Bernouilli)
		for i in 1:d["nobs"]
			@test d["response"][i] == 0 || d["response"][i] == 1
		end

	end

	@testset "Test Return value of likelihood" begin
	# Test some values of loglikelihood function

		# Test 1 : if y= [1 1..1] and betas = [0 0 .. 0] then loglik = log(cdf(0)) = log(1/2)
		d = Dict("coeff" => [1;1;1], "nobs" => 1, "data" => [1;2;3], "response" => [1], "distrib" => Normal(0,1))
		@test HW_unconstrained.loglik([0;0;0],d)==log(0.5)

	end

	@testset "Test return value of gradient" begin

		# Unit Test 1 :
		# X=[1;1;1], Y=1, beta=[1;1;1], n=1 => we test grad(1,1,1) = pdf(X*beta)/cdf(X*beta)
		betas = [1;1;1]
		n=1
		X = [1;1;1]
		y = [1]
		distrib = Normal(0,1)
		grad_true = [-(pdf(distrib, 3)/cdf(distrib, 3)); -(pdf(distrib, 3)/cdf(distrib, 3)); -(pdf(distrib, 3)/cdf(distrib, 3))]

		data = Dict("coeff"=>betas, "nobs"=>n, "data"=>X, "response"=>y, "distrib"=>distrib)
		grad_test = HW_unconstrained.grad!(betas, ones(3), data)
		@test_approx_eq grad_test grad_true

	end

end

################################################################################
@testset "test maximization results" begin

	@testset "maximize_like returns approximate result" begin

		using Optim
		srand(1234)
		result = HW_unconstrained.maximize_like()
		beta_result = result.minimizer
		d = HW_unconstrained.makeData()
		@test abs(beta_result[1] - d["coeff"][1]) < 0.5
		@test abs(beta_result[2] - d["coeff"][2]) < 0.5
		@test abs(beta_result[3] - d["coeff"][3]) < 0.5


	end

	@testset "maximize_grad returns accurate result" begin

		srand(1234)
		result = HW_unconstrained.maximize_like_grad()
		beta_result = result.minimizer
		d = HW_unconstrained.makeData()
		@test abs(beta_result[1] - d["coeff"][1]) < 0.5
		@test abs(beta_result[2] - d["coeff"][2]) < 0.5
		@test abs(beta_result[3] - d["coeff"][3]) < 0.5

	end

	#@testset "gradient is close to zero at max like estimate" begin

	#end

end
