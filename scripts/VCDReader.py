class VCDReader:

    def __init__(self, file_path, wire_names: set):
        self.wire_names = wire_names
        self.header = ""
        self.fin = open(file_path, 'r')
        self.name_to_id = {}
        self.blocks = []

        self.read_header()
        self.read_value_dumps()

    def read_header(self):
        print("Reading Header and Vars")
        for line in self.fin:
            if line == "$dumpvars\n":
                self.header += line
                break

            if line.startswith("$var"):
                if line.split()[4] not in self.wire_names:
                    continue
                else:
                    self.name_to_id[line.split()[4]] = line.split()[3]
            self.header += line

    def read_value_dumps(self):
        print("Reading value dumps")
        wire_ids = set(self.name_to_id.values())
        for line in self.fin:
            if line.startswith("#"):
                t = line.strip('\n#')
                self.blocks.append((t, []))
                continue

            block = self.blocks[-1]
            if line.startswith('b'):
                split = line.strip('\n').split()
                id, value = split[1], split[0].strip('b')
            else:
                id, value = line[1:-1], line[0]

            if id in wire_ids:
                block[1].append(line)
        
        self.blocks = [b for b in self.blocks if len(b[1]) > 0]

    def get_parsed_blocks(self):
        return [self.parse_block(b) for b in self.blocks]

    def parse_block(self, b):
        values = dict(self.parse_value(v) for v in b[1])
        return int(b[0]), values

    def parse_value(self, v):
        if v.startswith('b'):
            split = v.strip('\n').split()
            id, value = split[1], split[0].strip('b')
        else:
            id, value = v[1:-1], v[0]
        return id, value
        
