function test_block_structure(t, o, l, u)
    B = FEDVR(t, o)
    M = Matrix(undef, B)

    @test M isa BlockSkylineMatrix
    bs = M.block_sizes
    @test length(bs.l) == length(l)+1
    @test length(bs.u) == length(u)+1

    # TODO Test block sizes

    @test all(bs.l[1:end-2] .== l[1:end-1])
    length(l) > 0 && @test bs.l[end-1] ≥ l[end]
    @test all(bs.u[3:end] .== u[2:end])
    length(u) > 0 && @test bs.u[2] ≥ u[1]
end

@testset "Block structure" begin
    test_block_structure(1.0:2, [4], [], [])
    test_block_structure(1.0:7, [2,2,3,4,2,4], [1,1,2,1,2,1,1,1], [1,1,1,2,1,2,1,1])
    test_block_structure(1.0:8, [2,2,3,4,2,4,2], [1,1,2,1,2,1,1,2,1,1], [1,1,1,2,1,2,1,1,2,1])
    test_block_structure(range(0,1,length=7), 1 .+ (1:6), [1,2,1,2,1,2,1,2,1,1], [1,1,2,1,2,1,2,1,2,1])
    test_block_structure(range(0,1,length=7), reverse(1 .+ (1:6)), [1,2,1,2,1,2,1,2,1,1], [1,1,2,1,2,1,2,1,2,1])
    test_block_structure(range(0,1,length=7), 2 .+ (1:6), [1,2,1,2,1,2,1,2,1,1], [1,1,2,1,2,1,2,1,2,1])
    @test Matrix(undef, FEDVR(1.0:7, 2)) isa Tridiagonal
end

function test_blocks(f::Function, t, o)
    B = FEDVR(t, o)
    A = Matrix(undef, B)

    blocks = map(f, 1:FEDVRQuasi.nel(B))
    coords = vcat(1,1 .+ cumsum(B.order[1:end-1] .- 1))
    FEDVRQuasi.set_blocks!(f, A, B)

    for (i,(b,c)) in enumerate(zip(blocks,coords))
        sel = c .+ (0:size(b,1)-1)
        b′ = copy(A[sel,sel])
        i > 1 && (b′[1,1] -= blocks[i-1][end,end])
        i < length(blocks) && (b′[end,end] -= blocks[i+1][1,1])
        @test all(b′ .== b)
    end
end

@testset "Set blocks" begin
    test_blocks(1.0:3,[2,3]) do i
        i*ones(i+1,i+1)
    end
    begin
        o = [2,2,3,4,2,4]
        test_blocks(1.0:7,o) do i
            i*ones(o[i],o[i])
        end
    end
    begin
        o = 4 .+ (1:6)
        test_blocks(range(0,1,length=7), o) do i
            i*ones(o[i],o[i])
        end
    end
end
